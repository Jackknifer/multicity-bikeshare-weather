import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arviz as az
import pymc as pm

from audit import city_audit, integrity_report, roster_comparison
from baselines import BASELINES
from common import (
    WORK_ROOT,
    environment_snapshot,
    load_paths,
    read_json,
    verify_frozen_input,
    write_json,
)
from data import (
    CITY,
    DATE,
    load_daily,
    roster_prediction,
    roster_primary,
)
from metrics import WIS_SOURCE, add_scores, city_equal_summary, seasonal_naive_scale
from model import build_model, load_model_config, resolve_spec
from predict import block_forecast, one_step_forecast
from splits import (
    assert_no_leakage,
    fold_windows,
    load_splits,
    make_split_index,
    prepare_fold,
    prepare_full_period,
)

EVENT_LOG = None


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def emit(tag, payload):
    event = {"tag": tag, "time_utc": datetime.now(timezone.utc).isoformat(), "payload": payload}
    print(f"[{event['time_utc']}] {tag} {payload}", flush=True)
    if EVENT_LOG is not None:
        EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with EVENT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")


def run_directory(run_id, model_name, fold_name):
    return WORK_ROOT / "runs" / run_id / model_name / fold_name


def resolve_spec_for_run(config, model, weather_scale=1.0):
    spec = resolve_spec(config, model)
    spec["priors"] = dict(config["prior_scales"])
    if weather_scale != 1.0:
        for key in ("beta_temp_sd", "beta_temp_sq_sd", "beta_rain_sd",
                    "temp_moderator_sd", "moderator_sd"):
            spec["priors"][key] = spec["priors"][key] * weather_scale
    spec["weather_scale"] = weather_scale
    return spec


def command_audit(args):
    paths = load_paths(args.paths)
    frozen = verify_frozen_input(paths)
    frame = load_daily(paths["data_csv"])
    audit_frame = city_audit(frame)
    report = integrity_report(frame)
    comparison = roster_comparison(frame)
    out_dir = WORK_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(out_dir / "city_data_audit.csv", index=False)
    write_json(out_dir / "integrity_report.json", report)
    write_json(out_dir / "roster_comparison.json", comparison)
    write_json(out_dir / "frozen_input.json", frozen)
    emit("AUDIT", {
        "rows": report["rows"],
        "cities": report["cities"],
        "prediction_roster": len(report["prediction_roster"]),
        "primary_roster": len(report["primary_roster"]),
        "cities_with_missing_days": int((audit_frame["missing_days_in_span"] > 0).sum()),
    })
    return 0


def command_splits(args):
    paths = load_paths(args.paths)
    verify_frozen_input(paths)
    frame = load_daily(paths["data_csv"])
    payload = load_splits(args.splits)
    index = make_split_index(frame, payload)
    out_dir = WORK_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(out_dir / "split_index.csv", index=False)
    summary = {}
    for name in payload["folds"]:
        subset = index.loc[index["fold"] == name]
        summary[name] = {
            "train_rows": int((subset["split"] == "train").sum()),
            "test_rows": int((subset["split"] == "test").sum()),
            "after_cutoff_rows": int((subset["split"] == "after_cutoff").sum()),
        }
        bundle = prepare_fold(frame, payload, name)
        assert_no_leakage(bundle)
        emit("SPLIT", {"fold": name, **summary[name]})
    write_json(out_dir / "split_summary.json", summary)
    return 0


def _bfmi(idata):
    energy = idata.sample_stats["energy"].to_numpy()
    values = []
    for chain in range(energy.shape[0]):
        series = energy[chain]
        variance = float(np.var(series))
        if variance <= 0:
            raise ValueError("energy variance is not positive; BFMI is undefined")
        values.append(float(np.mean(np.diff(series) ** 2) / variance))
    return values


def _diagnostics(idata, extra):
    summary = az.summary(idata, round_to=8)
    sample_stats = idata.sample_stats
    bfmi = _bfmi(idata)
    diagnostics = {
        "parameters": int(len(summary)),
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "rhat_over_1_01": int((summary["r_hat"] > 1.01).sum()),
        "ess_bulk_under_400": int((summary["ess_bulk"] < 400).sum()),
        "ess_tail_under_400": int((summary["ess_tail"] < 400).sum()),
        "max_mcse_sd_ratio": float((summary["mcse_mean"] / summary["sd"]).max()),
        "divergences": int(sample_stats["diverging"].sum()),
        "bfmi": bfmi,
        "bfmi_min": float(np.min(bfmi)),
    }
    for key in ("depth", "maxdepth_reached", "tree_depth", "reached_max_treedepth"):
        if key in sample_stats:
            diagnostics[f"{key}_max"] = float(sample_stats[key].max())
            diagnostics[f"{key}_sum"] = int(sample_stats[key].sum())
    diagnostics.update(extra)
    diagnostics["passed"] = bool(
        diagnostics["max_rhat"] <= 1.01
        and diagnostics["min_ess_bulk"] >= 400
        and diagnostics["min_ess_tail"] >= 400
        and diagnostics["divergences"] == 0
        and diagnostics.get("maxdepth_reached_sum", 0) == 0
    )
    worst = summary.sort_values("r_hat", ascending=False).head(8)
    diagnostics["worst_rhat"] = {
        index: {
            "r_hat": float(row["r_hat"]),
            "ess_bulk": float(row["ess_bulk"]),
            "ess_tail": float(row["ess_tail"]),
        }
        for index, row in worst.iterrows()
    }
    return diagnostics, summary


def _model_spec_text(spec, config):
    lines = [
        f"# {spec['name']} model specification",
        "",
        spec["description"],
        "",
        f"- calendar terms: `{spec['calendar']}`",
        f"- annual harmonic pairs: {spec['harmonic_pairs']}",
        f"- city linear trend: {spec['trend']}",
        f"- AR(1) residual process: {spec['ar1']}",
        f"- climate moderators: {spec['weather_moderators']}",
        f"- weather coefficient pooling: `{spec['weather_pooling']}`",
        f"- city intercept parameterisation: `{'centred' if spec.get('alpha_centered', True) else 'non-centred'}`",
        "",
        "## Response",
        "",
        "`y_it = log(1 + usage_it)` on daily city observations, Student-t likelihood with a",
        "shared degrees-of-freedom parameter and city-specific scale.",
        "",
        "## Linear predictor",
        "",
        "`mu_it = alpha_i + b_temp_i * T_it + b_temp_sq_i * Q_it + b_rain_i * R_it + calendar + trend`",
        "",
        "- `T_it = (utci_mean_c_it - city training mean) / 10`",
        "- `Q_it = T_it^2 - city training mean of T^2`",
        "- `R_it = (log1p(precipitation_mm_it) - city training mean) / pooled training standard deviation`",
        "- calendar: shared sum-to-zero weekday effect (`weekday`), or a city weekend coefficient (`weekend`)",
        "- trend: city slope per 100 days, centred on the training-period mean day number",
        "",
        "All centring constants, scales and standard deviations are computed from training rows only.",
        "",
    ]
    if spec["ar1"]:
        lines.extend([
            "## AR(1) residual process",
            "",
            "`e_it = phi_i * e_i,t-1 + eta_it`, `eta_it ~ StudentT(nu, 0, sigma_i)`, `phi_i = tanh(z_i)`",
            "with a hierarchical prior on `z_i`. The first observation of every contiguous date",
            "segment uses a prescribed wide Student-t initial scale",
            "`sigma_init_i = sigma_i * sqrt(nu / ((nu - 2) * (1 - phi_i^2)))`, the variance-matched",
            "scale implied by a Gaussian stationary law. This is a moment-matched initial scale, not",
            "the exact stationary distribution of the Student-t innovation process. Segments are cut",
            "at any calendar gap larger than one day.",
            "",
            "`sigma_i` is the innovation scale, so it is not directly comparable to the residual",
            "scale of models without the AR term.",
            "",
        ])
    lines.extend([
        "## Prior scales",
        "",
        "| quantity | scale |",
        "|---|---:|",
    ])
    for key, value in config["prior_scales"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Weather coefficients are observational conditional associations given the calendar and",
        "trend controls. They are not causal effects.",
        "",
    ])
    return "\n".join(lines)


def command_fit(args):
    global EVENT_LOG
    paths = load_paths(args.paths)
    frozen = verify_frozen_input(paths)
    frame = load_daily(paths["data_csv"])
    splits = load_splits(args.splits)
    config = load_model_config(args.models_config)
    spec = resolve_spec_for_run(config, args.model, args.weather_scale)
    run_name = args.run_name or args.model

    bundle = _load_bundle(paths, splits, args.fold, args.roster)
    city_names = bundle["prep"]["city_names"]

    directory = run_directory(args.run_id, run_name, args.fold)
    directory.mkdir(parents=True, exist_ok=True)
    EVENT_LOG = directory / "events.jsonl"
    posterior_path = directory / "posterior.nc"
    if posterior_path.exists() and not args.resume:
        raise FileExistsError(f"posterior already exists: {posterior_path}; pass --resume to reuse")

    run_config = {
        "run_id": args.run_id,
        "run_name": run_name,
        "model": args.model,
        "fold": args.fold,
        "role": bundle["window"]["role"],
        "spec": {key: value for key, value in spec.items() if key != "priors"},
        "prior_scales": spec["priors"],
        "weather_prior_scale": args.weather_scale,
        "sampler": {
            "engine": "nutpie",
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "cores": args.cores,
            "seed": args.seed,
            "target_accept": args.target_accept,
            "maxdepth": args.maxdepth,
        },
        "window": {key: (str(value) if value is not None else None)
                   for key, value in bundle["window"].items()},
        "roster": args.roster,
        "train_rows": int(len(bundle["train_frame"])),
        "test_rows": int(len(bundle["test_frame"])) if bundle["test_frame"] is not None else 0,
        "cities": city_names,
        "frozen_input": frozen,
        "environment": environment_snapshot(),
        "validation_only": bool(args.validation_only),
        "wis_source": WIS_SOURCE,
    }
    write_json(directory / "config.json", run_config)
    write_json(directory / "preprocessing.json", bundle["prep"])
    (directory / "model_spec.md").write_text(
        _model_spec_text(spec, config), encoding="utf-8"
    )

    if posterior_path.exists() and args.resume:
        idata = az.from_netcdf(posterior_path)
        emit("RESUME", {"posterior": str(posterior_path)})
    else:
        model = build_model(spec, bundle["train_arrays"], city_names)
        import nutpie

        emit("START", {
            "model": args.model, "fold": args.fold, "train_rows": int(len(bundle["train_frame"])),
            "draws": args.draws, "tune": args.tune, "chains": args.chains,
        })
        start = time.time()
        compiled = nutpie.compile_pymc_model(model)
        idata = nutpie.sample(
            compiled,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            seed=args.seed,
            target_accept=args.target_accept,
            maxdepth=args.maxdepth,
            save_warmup=False,
            progress_bar=False,
        )
        elapsed = time.time() - start
        partial = directory / "posterior.partial.nc"
        idata.to_netcdf(partial)
        partial.replace(posterior_path)
        write_json(directory / "sampling_time.json", {"seconds": elapsed})
        emit("SAMPLED", {"model": args.model, "fold": args.fold, "seconds": round(elapsed, 1)})

    elapsed = read_json(directory / "sampling_time.json")["seconds"] if (
        directory / "sampling_time.json"
    ).exists() else None
    diagnostics, summary = _diagnostics(idata, {"model": args.model, "fold": args.fold,
                                                "sampling_seconds": elapsed,
                                                "validation_only": bool(args.validation_only)})
    write_json(directory / "diagnostics.json", diagnostics)
    summary.to_csv(directory / "posterior_summary.csv", index_label="parameter")
    emit("DIAGNOSTICS", {
        "model": args.model, "fold": args.fold, "passed": diagnostics["passed"],
        "max_rhat": diagnostics["max_rhat"], "min_ess_bulk": diagnostics["min_ess_bulk"],
        "min_ess_tail": diagnostics["min_ess_tail"], "divergences": diagnostics["divergences"],
        "maxdepth_reached_sum": diagnostics.get("maxdepth_reached_sum", 0),
    })
    return 0


def _load_bundle(paths, splits, fold, roster="prediction"):
    frame = load_daily(paths["data_csv"])
    if fold == "full":
        names = roster_primary(frame) if roster == "primary" else roster_prediction(frame)
        return prepare_full_period(frame, names)
    bundle = prepare_fold(frame, splits, fold)
    assert_no_leakage(bundle)
    return bundle


def command_predict(args):
    global EVENT_LOG

    paths = load_paths(args.paths)
    splits = load_splits(args.splits)
    config = load_model_config(args.models_config)
    spec = resolve_spec_for_run(config, args.model, getattr(args, "weather_scale", 1.0))
    run_name = getattr(args, "run_name", None) or args.model

    directory = run_directory(args.run_id, run_name, args.fold)
    EVENT_LOG = directory / "events.jsonl"
    posterior_path = directory / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"missing posterior: {posterior_path}")
    diagnostics = read_json(directory / "diagnostics.json")
    if not diagnostics["passed"] and not args.allow_failed_diagnostics:
        raise ValueError("diagnostics did not pass; refusing to publish predictions")

    bundle = _load_bundle(paths, splits, args.fold, getattr(args, "roster", "prediction"))
    if bundle["test_frame"] is None:
        raise ValueError("the full-period fit has no test partition and cannot be scored")
    idata = az.from_netcdf(posterior_path)

    frames = []
    extra = {}
    for task, function in (("block", block_forecast), ("one_step", one_step_forecast)):
        frame, task_diagnostics = function(idata, spec, bundle, sample_cap=args.sample_cap,
                                           paths=args.paths_count, seed=args.seed)
        frames.append(frame)
        for key, value in task_diagnostics.items():
            extra[f"{task}_{key}"] = value
    predictions = pd.concat(frames, ignore_index=True)
    predictions = add_scores(predictions)
    predictions.to_csv(directory / "predictions.csv", index=False)

    scale = seasonal_naive_scale(bundle["train_frame"])
    overall = {}
    per_city_records = []
    for task in ("block", "one_step"):
        subset = predictions.loc[predictions["task"] == task]
        summary, per_city = city_equal_summary(subset, scale=scale)
        overall[task] = summary
        per_city = per_city.reset_index()
        per_city["task"] = task
        per_city["model"] = args.model
        per_city["fold"] = args.fold
        per_city_records.append(per_city)
    metrics_city = pd.concat(per_city_records, ignore_index=True)
    metrics_city.to_csv(directory / "metrics_city.csv", index=False)

    overall_payload = {
        "model": args.model,
        "run_name": run_name,
        "fold": args.fold,
        "run_id": args.run_id,
        "tasks": overall,
        "negative_median_log_fraction": float((predictions["median_log"] < 0).mean()),
        "diagnostics_passed": diagnostics["passed"],
        "seasonal_naive_scale": scale,
        "predictive_sample_cap": args.sample_cap,
        "path_count": args.paths_count,
        "density_stability": extra,
    }
    write_json(directory / "metrics_overall.json", overall_payload)
    emit("METRICS", {
        "model": args.model, "fold": args.fold,
        "block_city_equal_wis": overall["block"]["city_equal_wis_log"],
        "block_city_equal_density": overall["block"]["city_equal_mean_log_predictive_density"],
        "one_step_city_equal_wis": overall["one_step"]["city_equal_wis_log"],
        "one_step_city_equal_density": overall["one_step"]["city_equal_mean_log_predictive_density"],
        "worst_city_block": overall["block"]["worst_city_wis"],
    })
    return 0


def command_baselines(args):
    global EVENT_LOG
    paths = load_paths(args.paths)
    splits = load_splits(args.splits)
    directory = WORK_ROOT / "runs" / args.run_id / "baselines" / args.fold
    directory.mkdir(parents=True, exist_ok=True)
    EVENT_LOG = directory / "events.jsonl"
    bundle = _load_bundle(paths, splits, args.fold, getattr(args, "roster", "prediction"))
    scale = seasonal_naive_scale(bundle["train_frame"])

    frames = []
    for name, function in BASELINES.items():
        frames.append(function(bundle))
    predictions = add_scores(pd.concat(frames, ignore_index=True))
    predictions.to_csv(directory / "predictions.csv", index=False)

    overall = {}
    per_city_records = []
    for model_name in BASELINES:
        for task in ("block", "one_step"):
            subset = predictions.loc[
                (predictions["model"] == model_name) & (predictions["task"] == task)
            ]
            summary, per_city = city_equal_summary(subset, scale=scale)
            overall[f"{model_name}|{task}"] = summary
            per_city = per_city.reset_index()
            per_city["task"] = task
            per_city["model"] = model_name
            per_city["fold"] = args.fold
            per_city_records.append(per_city)
    pd.concat(per_city_records, ignore_index=True).to_csv(
        directory / "metrics_city.csv", index=False
    )
    write_json(directory / "metrics_overall.json", {
        "fold": args.fold, "run_id": args.run_id, "models": overall,
        "seasonal_naive_scale": scale,
    })
    emit("BASELINES", {
        "fold": args.fold,
        "block": {name: round(overall[f"{name}|block"]["city_equal_wis_log"], 5)
                  for name in BASELINES},
    })
    return 0


PRIOR_QUANTILES = (0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995)


def command_prior_predictive(args):
    global EVENT_LOG
    paths = load_paths(args.paths)
    splits = load_splits(args.splits)
    config = load_model_config(args.models_config)
    spec = resolve_spec_for_run(config, args.model, getattr(args, "weather_scale", 1.0))
    run_name = getattr(args, "run_name", None) or args.model
    directory = run_directory(args.run_id, run_name, args.fold)
    directory.mkdir(parents=True, exist_ok=True)
    EVENT_LOG = directory / "events.jsonl"
    bundle = _load_bundle(paths, splits, args.fold, getattr(args, "roster", "prediction"))
    model = build_model(spec, bundle["train_arrays"], bundle["prep"]["city_names"])
    with model:
        prior = pm.sample_prior_predictive(draws=args.draws, random_seed=args.seed)

    def draws(name):
        if name not in prior.prior:
            return None
        return prior.prior[name].to_numpy().ravel()

    observed = bundle["train_arrays"]["response"]
    payload = {
        "model": args.model,
        "fold": args.fold,
        "draws": args.draws,
        "weather_prior_scale": spec.get("weather_scale", 1.0),
        "observed_log_usage_range": [float(observed.min()), float(observed.max())],
    }

    alpha = draws("alpha_city")
    if alpha is not None:
        payload["intercept_usage_quantiles"] = {
            str(q): float(np.quantile(np.expm1(alpha), q)) for q in PRIOR_QUANTILES
        }
        payload["intercept_negative_fraction"] = float(np.mean(alpha < 0.0))
    beta_temp = draws("beta_temp_city")
    beta_temp_sq = draws("beta_temp_sq_city")
    if beta_temp is not None:
        contrast = np.expm1(beta_temp + beta_temp_sq)
        payload["temp_10c_percent_quantiles"] = {
            str(q): float(np.quantile(100.0 * contrast, q)) for q in PRIOR_QUANTILES
        }
    beta_rain = draws("beta_rain_city")
    if beta_rain is not None:
        prep = bundle["prep"]
        rain_median = bundle["train_frame"].groupby(CITY)["precipitation_mm"].median()
        step = np.log1p(rain_median.to_numpy() + 10.0) - np.log1p(rain_median.to_numpy())
        contrast = np.expm1(beta_rain.reshape(-1, len(step)) * (step / prep["rain_pooled_sd"])[None, :])
        payload["rain_10mm_percent_quantiles"] = {
            str(q): float(np.quantile(100.0 * contrast, q)) for q in PRIOR_QUANTILES
        }
    beta_trend = draws("beta_trend_city")
    if beta_trend is not None:
        payload["trend_year_log_change_quantiles"] = {
            str(q): float(np.quantile(beta_trend * 3.65, q)) for q in PRIOR_QUANTILES
        }
    phi = draws("phi_city")
    if phi is not None:
        payload["phi_quantiles"] = {
            str(q): float(np.quantile(phi, q)) for q in PRIOR_QUANTILES
        }
    nu = draws("nu")
    if nu is not None:
        payload["nu_quantiles"] = {
            str(q): float(np.quantile(nu, q)) for q in PRIOR_QUANTILES
        }
    sigma = draws("sigma_city")
    if sigma is not None:
        payload["sigma_quantiles"] = {
            str(q): float(np.quantile(sigma, q)) for q in PRIOR_QUANTILES
        }
    write_json(directory / "prior_predictive.json", payload)
    emit("PRIOR_PREDICTIVE", {
        "model": args.model, "fold": args.fold,
        "rain_10mm_median": payload.get("rain_10mm_percent_quantiles", {}).get("0.5"),
        "temp_10c_median": payload.get("temp_10c_percent_quantiles", {}).get("0.5"),
        "phi_median": payload.get("phi_quantiles", {}).get("0.5"),
    })
    return 0


def command_effects(args):
    global EVENT_LOG

    paths = load_paths(args.paths)
    splits = load_splits(args.splits)
    config = load_model_config(args.models_config)
    spec = resolve_spec_for_run(config, args.model, getattr(args, "weather_scale", 1.0))
    run_name = getattr(args, "run_name", None) or args.model
    directory = run_directory(args.run_id, run_name, args.fold)
    EVENT_LOG = directory / "events.jsonl"
    posterior_path = directory / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"missing posterior: {posterior_path}")
    diagnostics = read_json(directory / "diagnostics.json")
    if not diagnostics["passed"] and not args.allow_failed_diagnostics:
        raise ValueError("diagnostics did not pass; refusing to publish effects")

    bundle = _load_bundle(paths, splits, args.fold, getattr(args, "roster", "prediction"))
    idata = az.from_netcdf(posterior_path)
    from effects import compute_effects

    payload = compute_effects(idata, spec, bundle, seed=args.seed)
    write_json(directory / "effects.json", payload)
    emit("EFFECTS", {
        "model": args.model, "fold": args.fold,
        "rain_10mm_city_equal_percent": payload["rain_10mm_city_equal_percent"],
        "temp_10c_city_equal_percent": payload["temp_10c_city_equal_percent"],
        "p_population_rain_negative": payload["p_population_rain_negative"],
    })
    return 0


def command_residuals(args):
    global EVENT_LOG

    paths = load_paths(args.paths)
    splits = load_splits(args.splits)
    config = load_model_config(args.models_config)
    spec = resolve_spec_for_run(config, args.model, getattr(args, "weather_scale", 1.0))
    run_name = getattr(args, "run_name", None) or args.model
    directory = run_directory(args.run_id, run_name, args.fold)
    EVENT_LOG = directory / "events.jsonl"
    posterior_path = directory / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"missing posterior: {posterior_path}")
    diagnostics = read_json(directory / "diagnostics.json")
    if not diagnostics["passed"] and not args.allow_failed_diagnostics:
        raise ValueError("diagnostics did not pass; refusing to publish residual diagnostics")

    bundle = _load_bundle(paths, splits, args.fold, getattr(args, "roster", "prediction"))
    idata = az.from_netcdf(posterior_path)
    from residuals import autocorrelation, posterior_predictive_check, residual_table

    frame, kind = residual_table(idata, spec, bundle, seed=args.seed)
    acf = autocorrelation(frame)
    frame.to_csv(directory / "residuals_train.csv", index=False)
    acf.to_csv(directory / "residual_acf.csv", index=False)

    ppc, negative_fraction = posterior_predictive_check(idata, spec, bundle, seed=args.seed)
    ppc.to_csv(directory / "ppc_train.csv", index=False)

    lag_one = acf.loc[acf["lag"] == 1, "autocorrelation"]
    payload = {
        "model": args.model,
        "fold": args.fold,
        "residual_kind": kind,
        "lag1_median": float(lag_one.median()),
        "lag1_min": float(lag_one.min()),
        "lag1_max": float(lag_one.max()),
        "cities_lag1_above_0_5": int((lag_one > 0.5).sum()),
        "training_coverage_90": float(ppc["covered90"].mean()),
        "negative_log_prediction_fraction": negative_fraction,
        "training_log_residual_rmse": float(np.sqrt(np.mean(ppc["residual_log"] ** 2))),
    }
    write_json(directory / "residual_summary.json", payload)
    emit("RESIDUALS", payload)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="STA5001 final project pipeline")
    parser.add_argument("--paths", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.set_defaults(func=command_audit)

    splits = subparsers.add_parser("splits")
    splits.add_argument("--splits", default=None)
    splits.set_defaults(func=command_splits)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--model", required=True)
    fit.add_argument("--fold", required=True)
    fit.add_argument("--run-id", required=True)
    fit.add_argument("--run-name", default=None)
    fit.add_argument("--weather-scale", type=float, default=1.0)
    fit.add_argument("--roster", default="prediction", choices=("prediction", "primary"))
    fit.add_argument("--splits", default=None)
    fit.add_argument("--models-config", default=None)
    fit.add_argument("--draws", type=int, default=2000)
    fit.add_argument("--tune", type=int, default=2000)
    fit.add_argument("--chains", type=int, default=4)
    fit.add_argument("--cores", type=int, default=4)
    fit.add_argument("--seed", type=int, default=27)
    fit.add_argument("--target-accept", type=float, default=0.92)
    fit.add_argument("--maxdepth", type=int, default=12)
    fit.add_argument("--resume", action="store_true")
    fit.add_argument("--validation-only", action="store_true")
    fit.set_defaults(func=command_fit)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--fold", required=True)
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--run-name", default=None)
    predict.add_argument("--weather-scale", type=float, default=1.0)
    predict.add_argument("--splits", default=None)
    predict.add_argument("--models-config", default=None)
    predict.add_argument("--sample-cap", type=int, default=400)
    predict.add_argument("--paths-count", type=int, default=200)
    predict.add_argument("--seed", type=int, default=27)
    predict.add_argument("--allow-failed-diagnostics", action="store_true")
    predict.set_defaults(func=command_predict)

    baselines = subparsers.add_parser("baselines")
    baselines.add_argument("--fold", required=True)
    baselines.add_argument("--run-id", required=True)
    baselines.add_argument("--splits", default=None)
    baselines.set_defaults(func=command_baselines)

    prior = subparsers.add_parser("prior-predictive")
    prior.add_argument("--model", required=True)
    prior.add_argument("--fold", required=True)
    prior.add_argument("--run-id", required=True)
    prior.add_argument("--run-name", default=None)
    prior.add_argument("--weather-scale", type=float, default=1.0)
    prior.add_argument("--roster", default="prediction", choices=("prediction", "primary"))
    prior.add_argument("--splits", default=None)
    prior.add_argument("--models-config", default=None)
    prior.add_argument("--draws", type=int, default=500)
    prior.add_argument("--seed", type=int, default=27)
    prior.set_defaults(func=command_prior_predictive)

    effects = subparsers.add_parser("effects")
    effects.add_argument("--model", required=True)
    effects.add_argument("--fold", required=True)
    effects.add_argument("--run-id", required=True)
    effects.add_argument("--run-name", default=None)
    effects.add_argument("--weather-scale", type=float, default=1.0)
    effects.add_argument("--roster", default="prediction", choices=("prediction", "primary"))
    effects.add_argument("--splits", default=None)
    effects.add_argument("--models-config", default=None)
    effects.add_argument("--seed", type=int, default=27)
    effects.add_argument("--allow-failed-diagnostics", action="store_true")
    effects.set_defaults(func=command_effects)

    residuals = subparsers.add_parser("residuals")
    residuals.add_argument("--model", required=True)
    residuals.add_argument("--fold", required=True)
    residuals.add_argument("--run-id", required=True)
    residuals.add_argument("--run-name", default=None)
    residuals.add_argument("--weather-scale", type=float, default=1.0)
    residuals.add_argument("--roster", default="prediction", choices=("prediction", "primary"))
    residuals.add_argument("--splits", default=None)
    residuals.add_argument("--models-config", default=None)
    residuals.add_argument("--seed", type=int, default=27)
    residuals.add_argument("--allow-failed-diagnostics", action="store_true")
    residuals.set_defaults(func=command_residuals)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
