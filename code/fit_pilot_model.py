from pathlib import Path
import argparse
import json

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "clean" / "bikeshare_weather_daily.csv"
OUTPUT_DIR = ROOT / "results" / "model"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def standardize(values):
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    require(sd > 0, "standard deviation must be positive")
    return (values - mean) / sd, mean, sd


def prepare_model_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    data = data.loc[data["continuous_system"] == 1].copy()
    require(data["city"].nunique() == 25, "primary sample must contain 25 cities")
    require(not data.duplicated(["city", "date"]).any(), "city-date keys must be unique")
    require(data.isna().sum().sum() == 0, "model data contain missing values")

    city_names = sorted(data["city"].unique())
    city_lookup = {city: index for index, city in enumerate(city_names)}
    data["city_index"] = data["city"].map(city_lookup).astype(int)
    data["temp_sq_centered"] = data["utci_anomaly_10c"] ** 2
    data["temp_sq_centered"] -= data.groupby("city")["temp_sq_centered"].transform("mean")

    rain_within = (data["precipitation_log1p"]
                   - data.groupby("city")["precipitation_log1p"].transform("mean"))
    rain_z, rain_mean, rain_sd = standardize(rain_within.to_numpy())
    data["rain_z"] = rain_z

    city_table = (
        data.groupby("city", sort=True)
        .agg(
            thermal_baseline=("utci_mean_c", "mean"),
            rain_baseline=("precipitation_mm", "mean"),
        )
        .reindex(city_names)
    )
    thermal_z, thermal_mean, thermal_sd = standardize(city_table["thermal_baseline"].to_numpy())
    rain_baseline_z, rain_baseline_mean, rain_baseline_sd = standardize(
        np.log1p(city_table["rain_baseline"].to_numpy())
    )

    arrays = {
        "city_index": data["city_index"].to_numpy(),
        "temp": data["utci_anomaly_10c"].to_numpy(),
        "temp_sq": data["temp_sq_centered"].to_numpy(),
        "rain": data["rain_z"].to_numpy(),
        "weekend": data["weekend"].to_numpy(),
        "season_sin": data["season_sin"].to_numpy(),
        "season_cos": data["season_cos"].to_numpy(),
        "response": data["usage_log1p"].to_numpy(),
        "thermal_baseline": thermal_z,
        "rain_baseline": rain_baseline_z,
    }
    scaling = {
        "rain_within_log1p_mean": rain_mean,
        "rain_within_log1p_sd": rain_sd,
        "thermal_baseline_mean_c": thermal_mean,
        "thermal_baseline_sd_c": thermal_sd,
        "rain_baseline_log1p_mean": rain_baseline_mean,
        "rain_baseline_log1p_sd": rain_baseline_sd,
    }
    return data, city_names, arrays, scaling


def varying_coefficient(name, mean, scale, moderator=None, moderator_scale=0.3):
    population = pm.Normal(f"{name}_population", mean, scale)
    if moderator is None:
        center = population
    else:
        moderator_effect = pm.Normal(f"{name}_moderator", 0.0, moderator_scale)
        center = population + moderator_effect * moderator
    group_sd = pm.HalfNormal(f"{name}_city_sd", scale)
    offset = pm.Normal(f"{name}_offset", 0.0, 1.0, dims="city")
    return pm.Deterministic(f"{name}_city", center + group_sd * offset, dims="city")


def build_model(city_names, arrays):
    coords = {
        "city": city_names,
        "observation": np.arange(len(arrays["response"])),
    }
    with pm.Model(coords=coords) as model:
        city_index = pm.Data("city_index", arrays["city_index"], dims="observation")
        temp = pm.Data("temp", arrays["temp"], dims="observation")
        temp_sq = pm.Data("temp_sq", arrays["temp_sq"], dims="observation")
        rain = pm.Data("rain", arrays["rain"], dims="observation")
        weekend = pm.Data("weekend", arrays["weekend"], dims="observation")
        season_sin = pm.Data("season_sin", arrays["season_sin"], dims="observation")
        season_cos = pm.Data("season_cos", arrays["season_cos"], dims="observation")
        thermal_baseline = pm.Data("thermal_baseline", arrays["thermal_baseline"], dims="city")
        rain_baseline = pm.Data("rain_baseline", arrays["rain_baseline"], dims="city")

        alpha = varying_coefficient("alpha", 7.0, 2.0)
        beta_temp = varying_coefficient(
            "beta_temp", 0.0, 0.5, moderator=thermal_baseline
        )
        beta_temp_sq = varying_coefficient(
            "beta_temp_sq", 0.0, 0.3, moderator=thermal_baseline, moderator_scale=0.2
        )
        beta_rain = varying_coefficient(
            "beta_rain", 0.0, 0.3, moderator=rain_baseline, moderator_scale=0.2
        )
        beta_weekend = varying_coefficient("beta_weekend", 0.0, 0.4)
        beta_sin = varying_coefficient("beta_sin", 0.0, 0.8)
        beta_cos = varying_coefficient("beta_cos", 0.0, 0.8)

        log_sigma_population = pm.Normal("log_sigma_population", -0.5, 0.8)
        log_sigma_city_sd = pm.HalfNormal("log_sigma_city_sd", 0.5)
        log_sigma_offset = pm.Normal("log_sigma_offset", 0.0, 1.0, dims="city")
        sigma_city = pm.Deterministic(
            "sigma_city",
            pm.math.exp(log_sigma_population + log_sigma_city_sd * log_sigma_offset),
            dims="city",
        )
        nu_minus_two = pm.Exponential("nu_minus_two", 0.1)
        nu = pm.Deterministic("nu", nu_minus_two + 2.0)

        mu = (
            alpha[city_index]
            + beta_temp[city_index] * temp
            + beta_temp_sq[city_index] * temp_sq
            + beta_rain[city_index] * rain
            + beta_weekend[city_index] * weekend
            + beta_sin[city_index] * season_sin
            + beta_cos[city_index] * season_cos
        )
        pm.StudentT(
            "observed_log_usage",
            nu=nu,
            mu=mu,
            sigma=sigma_city[city_index],
            observed=arrays["response"],
            dims="observation",
        )
    return model


def build_diagnostics(idata, data, city_names, scaling, draws, tune, chains):
    """Summarise a fitted pilot run.

    Two scopes are reported and kept apart by name. ``*_selected`` covers the
    eight population-level hyperparameters quoted in the report; ``*_all``
    covers every parameter component in the posterior, which is what a reader
    has to look at before claiming the run converged.
    """
    summary_names = [
        "beta_temp_population",
        "beta_temp_moderator",
        "beta_temp_sq_population",
        "beta_temp_sq_moderator",
        "beta_rain_population",
        "beta_rain_moderator",
        "beta_weekend_population",
        "nu",
    ]
    summary = az.summary(idata, var_names=summary_names, round_to=6)
    full = az.summary(idata, var_names=None, round_to=6)

    divergences = int(idata.sample_stats["diverging"].sum().item())
    depth = idata.sample_stats["depth"]
    maxdepth = idata.sample_stats["maxdepth_reached"]

    return {
        "pymc_version": pm.__version__,
        "rows": int(len(data)),
        "cities": int(len(city_names)),
        "draws": int(draws),
        "tune": int(tune),
        "chains": int(chains),
        "divergences": divergences,
        "max_rhat_selected": float(summary["r_hat"].max()),
        "min_ess_bulk_selected": float(summary["ess_bulk"].min()),
        "min_ess_tail_selected": float(summary["ess_tail"].min()),
        "parameters": int(len(full)),
        "max_rhat_all": float(full["r_hat"].max()),
        "min_ess_bulk_all": float(full["ess_bulk"].min()),
        "min_ess_tail_all": float(full["ess_tail"].min()),
        "rhat_above_threshold": int((full["r_hat"] > 1.01).sum()),
        "max_tree_depth": int(depth.max().item()),
        "maxdepth_reached": int(maxdepth.sum().item()),
        "maxdepth_reached_rate": float(maxdepth.mean().item()),
        "scaling": scaling,
        "summary": {
            index: {column: float(value) for column, value in row.items()}
            for index, row in summary.to_dict(orient="index").items()
        },
    }


def write_diagnostics(diagnostics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "pilot_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )


def recompute_diagnostics():
    """Rebuild pilot_diagnostics.json from the stored netcdf without sampling."""
    model_path = OUTPUT_DIR / "pilot_model.nc"
    require(model_path.is_file(), f"missing file: {model_path}")
    data, city_names, arrays, scaling = prepare_model_data()
    idata = az.from_netcdf(model_path)
    draws = int(idata.posterior.sizes["draw"])
    chains = int(idata.posterior.sizes["chain"])
    tune = idata.posterior.attrs.get("tuning_steps")
    require(tune is not None, "stored run does not record its tuning steps")
    diagnostics = build_diagnostics(
        idata, data, city_names, scaling,
        draws=draws, tune=int(tune), chains=chains,
    )
    write_diagnostics(diagnostics)
    print(json.dumps(diagnostics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--tune", type=int, default=200)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument(
        "--recompute-diagnostics",
        action="store_true",
        help="rebuild pilot_diagnostics.json from the stored netcdf instead of sampling",
    )
    args = parser.parse_args()

    if args.recompute_diagnostics:
        recompute_diagnostics()
        return

    require(args.draws > 0, "draws must be positive")
    require(args.tune > 0, "tune must be positive")
    require(args.chains >= 2, "at least two chains are required")
    data, city_names, arrays, scaling = prepare_model_data()
    model = build_model(city_names, arrays)

    with model:
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=min(args.chains, 4),
            random_seed=27,
            nuts_sampler="nutpie",
            target_accept=0.92,
            progressbar=False,
        )

    diagnostics = build_diagnostics(
        idata, data, city_names, scaling,
        draws=args.draws, tune=args.tune, chains=args.chains,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(OUTPUT_DIR / "pilot_model.nc")
    write_diagnostics(diagnostics)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
