import json
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

MODELS_FILE = Path(__file__).resolve().parents[1] / "configs" / "models.json"

WEATHER_NAMES = ("beta_temp", "beta_temp_sq", "beta_rain")


def sum_to_zero_basis(size):
    basis = np.zeros((size, size - 1))
    for column in range(size - 1):
        basis[: column + 1, column] = 1.0
        basis[column + 1, column] = -(column + 1)
        basis[:, column] /= np.linalg.norm(basis[:, column])
    return basis


def load_model_config(path=None):
    target = Path(path) if path else MODELS_FILE
    payload = json.loads(target.read_text(encoding="utf-8"))
    for key in ("prior_scales", "models"):
        if key not in payload:
            raise ValueError(f"model configuration missing {key}")
    return payload


def resolve_spec(config, name):
    if name not in config["models"]:
        raise KeyError(f"unknown model specification: {name}")
    spec = dict(config["models"][name])
    spec["name"] = name
    return spec


def varying_coefficient(name, mean, scale, priors, moderator=None, moderator_scale=None,
                        pooling="hierarchical", centered=False):
    population = pm.Normal(f"{name}_population", mean, scale)
    if moderator is None:
        center = population
    else:
        moderator_effect = pm.Normal(f"{name}_moderator", 0.0, moderator_scale)
        center = population + moderator_effect * moderator
    if pooling == "hierarchical":
        group_sd = pm.HalfNormal(f"{name}_city_sd", scale)
        if centered:
            city = pm.Normal(f"{name}_city", center, group_sd, dims="city")
        else:
            offset = pm.Normal(f"{name}_offset", 0.0, 1.0, dims="city")
            city = pm.Deterministic(f"{name}_city", center + group_sd * offset, dims="city")
    elif pooling == "independent":
        fixed_sd = scale * np.sqrt(2.0)
        city = pm.Normal(f"{name}_city", center, fixed_sd, dims="city")
    else:
        raise ValueError(f"unknown pooling scheme: {pooling}")
    return city


def build_model(spec, arrays, city_names, coords=None):
    priors = spec["priors"]
    pooling = spec["weather_pooling"]
    observations = len(arrays["response"])
    if coords is None:
        coords = {"city": list(city_names), "observation": np.arange(observations)}
    coords = dict(coords)
    coords.setdefault("weekday", np.arange(7))
    coords.setdefault("weekday_contrast", np.arange(6))

    with pm.Model(coords=coords) as model:
        city_index = pm.Data("city_index", arrays["city_index"], dims="observation")
        temp = pm.Data("temp", arrays["temp"], dims="observation")
        temp_sq = pm.Data("temp_sq", arrays["temp_sq"], dims="observation")
        rain = pm.Data("rain", arrays["rain"], dims="observation")
        season_sin = pm.Data("season_sin", arrays["season_sin"], dims="observation")
        season_cos = pm.Data("season_cos", arrays["season_cos"], dims="observation")
        weekday = pm.Data("weekday_index", arrays["weekday"], dims="observation")
        weekend = pm.Data("weekend", arrays["weekend"], dims="observation")
        trend = pm.Data("trend", arrays["trend"], dims="observation")
        response = pm.Data("response", arrays["response"], dims="observation")
        thermal_baseline = pm.Data("thermal_baseline", arrays["thermal_baseline"], dims="city")
        rain_baseline = pm.Data("rain_baseline", arrays["rain_baseline"], dims="city")

        alpha = varying_coefficient(
            "alpha", 7.0, priors["alpha_population_sd"], priors, pooling="hierarchical",
            centered=bool(spec.get("alpha_centered", True)),
        )
        if spec["weather_moderators"]:
            beta_temp = varying_coefficient(
                "beta_temp", 0.0, priors["beta_temp_sd"], priors,
                moderator=thermal_baseline, moderator_scale=priors["temp_moderator_sd"],
                pooling=pooling,
            )
            beta_temp_sq = varying_coefficient(
                "beta_temp_sq", 0.0, priors["beta_temp_sq_sd"], priors,
                moderator=thermal_baseline, moderator_scale=priors["temp_moderator_sd"],
                pooling=pooling,
            )
            beta_rain = varying_coefficient(
                "beta_rain", 0.0, priors["beta_rain_sd"], priors,
                moderator=rain_baseline, moderator_scale=priors["moderator_sd"],
                pooling=pooling,
            )
        else:
            beta_temp = varying_coefficient(
                "beta_temp", 0.0, priors["beta_temp_sd"], priors, pooling=pooling
            )
            beta_temp_sq = varying_coefficient(
                "beta_temp_sq", 0.0, priors["beta_temp_sq_sd"], priors, pooling=pooling
            )
            beta_rain = varying_coefficient(
                "beta_rain", 0.0, priors["beta_rain_sd"], priors, pooling=pooling
            )

        mu = (
            alpha[city_index]
            + beta_temp[city_index] * temp
            + beta_temp_sq[city_index] * temp_sq
            + beta_rain[city_index] * rain
        )

        if spec["calendar"] == "weekend":
            beta_weekend = varying_coefficient(
                "beta_weekend", 0.0, priors["beta_weekend_sd"], priors
            )
            mu = mu + beta_weekend[city_index] * weekend
        elif spec["calendar"] == "weekday":
            basis = pm.Data("weekday_basis", sum_to_zero_basis(7))
            weekday_theta = pm.Normal(
                "weekday_theta", 0.0, priors["beta_weekday_sd"], dims="weekday_contrast"
            )
            weekday_effect = pm.Deterministic(
                "weekday_effect", pt.dot(basis, weekday_theta), dims="weekday"
            )
            mu = mu + weekday_effect[weekday]
        else:
            raise ValueError(f"unknown calendar specification: {spec['calendar']}")

        if spec["harmonic_pairs"] >= 1:
            beta_sin = varying_coefficient("beta_sin", 0.0, priors["beta_season_sd"], priors)
            beta_cos = varying_coefficient("beta_cos", 0.0, priors["beta_season_sd"], priors)
            mu = mu + beta_sin[city_index] * season_sin + beta_cos[city_index] * season_cos

        if spec["trend"]:
            beta_trend = varying_coefficient(
                "beta_trend", 0.0, priors["beta_trend_population_sd"], priors,
                pooling=pooling,
            )
            mu = mu + beta_trend[city_index] * trend

        log_sigma_population = pm.Normal(
            "log_sigma_population", priors["log_sigma_population_mean"],
            priors["log_sigma_population_sd"],
        )
        log_sigma_city_sd = pm.HalfNormal("log_sigma_city_sd", priors["log_sigma_city_sd"])
        log_sigma_city = pm.Normal(
            "log_sigma_city", log_sigma_population, log_sigma_city_sd, dims="city"
        )
        sigma_city = pm.Deterministic(
            "sigma_city", pm.math.exp(log_sigma_city), dims="city"
        )
        nu = pm.Deterministic("nu", pm.Exponential("nu_minus_two", priors["nu_minus_two_rate"]) + 2.0)

        if spec["ar1"]:
            z_population = pm.Normal("z_population", 0.0, priors["z_population_sd"])
            z_city_sd = pm.HalfNormal("z_city_sd", priors["z_city_sd"])
            z_offset = pm.Normal("z_offset", 0.0, 1.0, dims="city")
            z_city = pm.Deterministic("z_city", z_population + z_city_sd * z_offset, dims="city")
            phi_city = pm.Deterministic("phi_city", pt.tanh(z_city), dims="city")
            variance_ratio = nu / (nu - 2.0)
            sigma_init = pm.Deterministic(
                "sigma_init",
                sigma_city * pt.sqrt(variance_ratio / (1.0 - phi_city ** 2)),
                dims="city",
            )
            restart = pm.Data("restart", arrays["restart"], dims="observation")

            def ar_step(y_t, mu_t, phi_t, restart_t, previous_error):
                error_t = y_t - mu_t
                offset_t = pt.switch(pt.eq(restart_t, 1.0), pt.zeros_like(error_t), phi_t * previous_error)
                return error_t, offset_t

            (_, offset_seq), _ = scan(
                ar_step,
                sequences=[response, mu, phi_city[city_index], restart],
                outputs_info=[pt.as_tensor_variable(np.float64(0.0)), None],
            )
            sigma_observation = pt.switch(
                pt.eq(restart, 1.0), sigma_init[city_index], sigma_city[city_index]
            )
            conditional = pm.StudentT.dist(
                nu=nu, mu=mu + offset_seq, sigma=sigma_observation
            )
            pm.Potential("ar1_log_likelihood", pt.sum(pm.logp(conditional, response)))
        else:
            pm.StudentT(
                "observed_log_usage",
                nu=nu,
                mu=mu,
                sigma=sigma_city[city_index],
                observed=arrays["response"],
                dims="observation",
            )
    return model


def coefficient_names(spec):
    names = ["alpha_city"]
    names.extend(f"{name}_city" for name in WEATHER_NAMES)
    if spec["calendar"] == "weekend":
        names.append("beta_weekend_city")
    if spec["harmonic_pairs"] >= 1:
        names.extend(["beta_sin_city", "beta_cos_city"])
    if spec["trend"]:
        names.append("beta_trend_city")
    return names


def flat_posterior(idata, name):
    return (
        idata.posterior[name]
        .stack(sample=("chain", "draw"))
        .transpose("sample", ...)
        .to_numpy()
    )


def select_samples(idata, cap, seed):
    total = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    if total <= cap:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=cap, replace=False))


def linear_predictor(idata, spec, arrays, take):
    city_index = arrays["city_index"]
    mu = flat_posterior(idata, "alpha_city")[take][:, city_index]
    for name, key in (
        ("beta_temp_city", "temp"),
        ("beta_temp_sq_city", "temp_sq"),
        ("beta_rain_city", "rain"),
    ):
        mu = mu + flat_posterior(idata, name)[take][:, city_index] * arrays[key][None, :]
    if spec["calendar"] == "weekend":
        mu = mu + flat_posterior(idata, "beta_weekend_city")[take][:, city_index] * arrays["weekend"][None, :]
    elif spec["calendar"] == "weekday":
        mu = mu + flat_posterior(idata, "weekday_effect")[take][:, arrays["weekday"]]
    if spec["harmonic_pairs"] >= 1:
        mu = mu + flat_posterior(idata, "beta_sin_city")[take][:, city_index] * arrays["season_sin"][None, :]
        mu = mu + flat_posterior(idata, "beta_cos_city")[take][:, city_index] * arrays["season_cos"][None, :]
    if spec["trend"]:
        mu = mu + flat_posterior(idata, "beta_trend_city")[take][:, city_index] * arrays["trend"][None, :]
    return mu
