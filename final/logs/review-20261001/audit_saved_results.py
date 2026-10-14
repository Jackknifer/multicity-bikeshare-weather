import copy
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats
from pytensor.graph.traversal import ancestors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import load_paths, verify_frozen_input
from data import load_daily, roster_primary
from model import build_model, flat_posterior, linear_predictor, select_samples
from predict import block_forecast, one_step_forecast
from run import resolve_spec_for_run
from model import load_model_config
from splits import load_splits, prepare_fold, prepare_full_period, assert_no_leakage

RUNS = ROOT / "runs" / "formal_20260930"
OUT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def independently_score(frame):
    observed = frame["observed_log"].to_numpy()
    score = 0.5 * np.abs(observed - frame["median_log"].to_numpy())
    for level in (50, 80, 90, 95):
        alpha = 1 - level / 100
        lower = frame[f"lower_{level}"].to_numpy()
        upper = frame[f"upper_{level}"].to_numpy()
        assert np.all(lower <= upper)
        assert np.all(lower <= frame["median_log"])
        assert np.all(upper >= frame["median_log"])
        score += alpha / 2 * (upper - lower)
        score += np.maximum(lower - observed, 0)
        score += np.maximum(observed - upper, 0)
    score /= 4.5
    assert np.allclose(score, frame["wis_log"], atol=1e-12)
    assert np.allclose(np.log1p(frame["observed_usage"]), observed)
    assert np.allclose(np.expm1(frame["median_log"]), frame["median_usage"])
    city = frame.assign(recomputed_wis=score).groupby("city")
    return {
        "city_equal_wis_log": float(city["recomputed_wis"].mean().mean()),
        "city_equal_mean_log_predictive_density": float(city["log_predictive_density"].mean().mean()),
        "city_equal_mae_log": float(frame.assign(error=np.abs(observed - frame["median_log"])).groupby("city")["error"].mean().mean()),
        "city_equal_coverage_90": float(frame.assign(covered=(observed >= frame["lower_90"]) & (observed <= frame["upper_90"])).groupby("city")["covered"].mean().mean()),
    }


def audit_metrics(data, splits):
    records = []
    for path in sorted(RUNS.glob("*/*/predictions.csv")):
        saved = pd.read_csv(path, parse_dates=["date"])
        saved_metrics = read_json(path.with_name("metrics_overall.json"))
        bundle = prepare_fold(data, splits, path.parent.name)
        for (model, task), group in saved.groupby(["model", "task"]):
            assert len(group) == 23 * 56
            assert not group.duplicated(["city", "date"]).any()
            joined = group.merge(bundle["test_frame"][["city", "date", "usage"]], on=["city", "date"], validate="one_to_one")
            assert len(joined) == 23 * 56
            assert np.allclose(joined["usage"], joined["observed_usage"])
            scored = independently_score(group)
            expected = saved_metrics["tasks"][task] if "tasks" in saved_metrics else saved_metrics["models"][f"{model}|{task}"]
            for key, value in scored.items():
                assert np.isclose(value, expected[key], atol=1e-12), (path, key)
            records.append({"run_name": path.parent.parent.name, "fold": path.parent.name, "model": model, "task": task, **scored})
    result = pd.DataFrame(records)
    result.to_csv(OUT / "recomputed_metrics.csv", index=False)
    print("预测指标核验", len(result), "组全部通过", flush=True)
    return records


def audit_posteriors(data, splits):
    records = []
    for path in sorted(RUNS.glob("*/*/posterior.nc")):
        directory = path.parent
        config = read_json(directory / "config.json")
        saved = read_json(directory / "diagnostics.json")
        prep = read_json(directory / "preprocessing.json")
        idata = az.from_netcdf(path)
        assert idata.posterior.sizes["chain"] == 4
        assert idata.posterior.sizes["draw"] == config["sampler"]["draws"]
        assert list(idata.posterior.coords["city"].values) == config["cities"]
        bundle = prepare_full_period(data, roster_primary(data)) if config["fold"] == "full" else prepare_fold(data, splits, config["fold"])
        assert prep == bundle["prep"]
        assert config["train_rows"] == len(bundle["train_frame"])
        if "constant_data" in idata:
            for key in ("response", "temp", "temp_sq", "rain", "trend", "city_index", "restart"):
                if key in idata.constant_data:
                    assert np.allclose(idata.constant_data[key].to_numpy(), bundle["train_arrays"][key]), (directory, key)
        summary = az.summary(idata, round_to=8)
        diagnostics = {
            "max_rhat": float(summary["r_hat"].max()),
            "min_ess_bulk": float(summary["ess_bulk"].min()),
            "min_ess_tail": float(summary["ess_tail"].min()),
            "divergences": int(idata.sample_stats["diverging"].sum()),
        }
        for key, value in diagnostics.items():
            assert np.isclose(value, saved[key], atol=1e-6), (directory, key, value, saved[key])
        record = {"run_name": directory.parent.name, "fold": directory.name, "draws": idata.posterior.sizes["draw"], "constant_data_available": "constant_data" in idata, "constant_data_fields": list(idata.constant_data.data_vars) if "constant_data" in idata else [], **diagnostics}
        if config["spec"]["ar1"]:
            nu = flat_posterior(idata, "nu")
            record["initial_variance_inflation_median"] = float(np.median(nu / (nu - 2)))
            record["initial_variance_inflation_95"] = np.quantile(nu / (nu - 2), [0.025, 0.975]).tolist()
        if (directory / "effects.json").exists():
            effects = read_json(directory / "effects.json")
            take = select_samples(idata, effects["sample_cap"], 27)
            rain = flat_posterior(idata, "beta_rain_city")[take]
            names = prep["city_names"]
            rain_median = bundle["train_frame"].groupby("city")["precipitation_mm"].median().reindex(names).to_numpy()
            delta = (np.log1p(rain_median + 10) - np.log1p(rain_median)) / prep["rain_pooled_sd"]
            effect = 100 * np.expm1(rain * delta).mean(axis=1)
            for key, value in zip(("q025", "median", "q975"), np.quantile(effect, [0.025, 0.5, 0.975])):
                assert np.isclose(value, effects["rain_10mm_city_equal_percent"][key])
            record["rain_cities_interval_includes_zero"] = [item["city"] for item in effects["cities"] if item["rain_10mm_percent"]["q025"] <= 0 <= item["rain_10mm_percent"]["q975"]]
        records.append(record)
        print("后验与配置核验", directory.parent.name, directory.name, diagnostics, flush=True)
        idata.close()
    return records


def audit_real_prediction_isolation(data, splits, model_config):
    directory = RUNS / "M2" / "dev1"
    idata = az.from_netcdf(directory / "posterior.nc")
    spec = resolve_spec_for_run(model_config, "M2")
    bundle = prepare_fold(data, splits, "dev1")
    assert_no_leakage(bundle)
    original, _ = block_forecast(idata, spec, bundle, sample_cap=64, paths=20)
    changed = copy.deepcopy(bundle)
    changed["test_frame"]["usage"] += 100
    changed["test_arrays"]["response"] = np.log1p(changed["test_frame"]["usage"].to_numpy())
    revised, _ = block_forecast(idata, spec, changed, sample_cap=64, paths=20)
    columns = ["median_log", "lower_50", "upper_50", "lower_90", "upper_90"]
    assert np.allclose(original[columns], revised[columns])
    base_step, _ = one_step_forecast(idata, spec, bundle, sample_cap=64, paths=20)
    selected_date = bundle["test_frame"]["date"].min() + pd.Timedelta(days=10)
    altered = copy.deepcopy(bundle)
    positions = np.flatnonzero(altered["test_frame"]["date"].to_numpy() == selected_date)
    altered["test_frame"].loc[positions, "usage"] += 100
    altered["test_arrays"]["response"][positions] = np.log1p(altered["test_frame"].loc[positions, "usage"])
    revised_step, _ = one_step_forecast(idata, spec, altered, sample_cap=64, paths=20)
    before_or_same = base_step["date"] <= selected_date
    assert np.allclose(base_step.loc[before_or_same, columns], revised_step.loc[before_or_same, columns])
    next_day = base_step["date"] == selected_date + pd.Timedelta(days=1)
    assert not np.allclose(base_step.loc[next_day, "median_log"], revised_step.loc[next_day, "median_log"])
    later = base_step["date"] > selected_date + pd.Timedelta(days=1)
    assert np.allclose(base_step.loc[later, columns], revised_step.loc[later, columns])
    revised_prep = data.copy()
    mask = revised_prep["date"] > bundle["window"]["cutoff"]
    revised_prep.loc[mask, "utci_mean_c"] += 20
    revised_prep.loc[mask, "precipitation_log1p"] += 2
    assert prepare_fold(revised_prep, splits, "dev1")["prep"] == bundle["prep"]
    idata.close()
    print("实际后验预测隔离及训练期标准化核验通过", flush=True)
    return {"block_ignores_test_usage": True, "one_step_only_uses_previous_usage": True, "preprocessing_training_only": True, "posterior_source": str(directory / "posterior.nc"), "sample_cap": 64, "paths": 20}


def audit_ablation(data, splits, config):
    bundle = prepare_fold(data, splits, "dev1")
    result = {}
    for name in ("A0", "A1"):
        spec = resolve_spec_for_run(config, name)
        model = build_model(spec, bundle["train_arrays"], bundle["prep"]["city_names"])
        parents = {}
        for variable in ("beta_rain_city", "beta_trend_city"):
            rv = model.named_vars[variable]
            dependencies = [node.name for node in ancestors([rv]) if node.name]
            parents[variable] = sorted(set(dependencies))
        result[name] = parents
    print("汇聚比较的参数依赖", result, flush=True)
    return result


def main():
    paths = load_paths()
    frozen = verify_frozen_input(paths)
    data = load_daily(paths["data_csv"])
    splits = load_splits()
    config = load_model_config()
    results = {"frozen_input": frozen, "metrics": audit_metrics(data, splits)}
    results["prediction_isolation"] = audit_real_prediction_isolation(data, splits, config)
    results["ablation_dependencies"] = audit_ablation(data, splits, config)
    results["posteriors"] = audit_posteriors(data, splits)
    (OUT / "independent_checks.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("独立核验完成", flush=True)


if __name__ == "__main__":
    main()
