import numpy as np
import pandas as pd

from model import flat_posterior, select_samples

EFFECT_SAMPLE_CAP = 4000


def interval(values):
    return {
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def compute_effects(idata, spec, bundle, seed=27, sample_cap=EFFECT_SAMPLE_CAP):
    take = select_samples(idata, sample_cap, seed)
    names = bundle["prep"]["city_names"]
    train_frame = bundle["train_frame"]
    prep = bundle["prep"]

    rain_sd = prep["rain_pooled_sd"]
    rain_median = train_frame.groupby("city")["precipitation_mm"].median().reindex(names).to_numpy()
    rain_q90 = train_frame.groupby("city")["precipitation_mm"].quantile(0.90).reindex(names).to_numpy()
    temp_mean = train_frame.groupby("city")["utci_mean_c"].mean().reindex(names).to_numpy()
    temp_q25 = train_frame.groupby("city")["utci_mean_c"].quantile(0.25).reindex(names).to_numpy()
    temp_q75 = train_frame.groupby("city")["utci_mean_c"].quantile(0.75).reindex(names).to_numpy()
    temp_min = train_frame.groupby("city")["utci_mean_c"].min().reindex(names).to_numpy()
    temp_max = train_frame.groupby("city")["utci_mean_c"].max().reindex(names).to_numpy()

    beta_rain = flat_posterior(idata, "beta_rain_city")[take]
    beta_temp = flat_posterior(idata, "beta_temp_city")[take]
    beta_temp_sq = flat_posterior(idata, "beta_temp_sq_city")[take]

    rain_step = (np.log1p(rain_median + 10.0) - np.log1p(rain_median)) / rain_sd
    rain_support = (np.log1p(rain_q90) - np.log1p(rain_median)) / rain_sd
    temp_support_t25 = (temp_q25 - temp_mean) / 10.0
    temp_support_t75 = (temp_q75 - temp_mean) / 10.0
    temp_support_delta = temp_support_t75 - temp_support_t25
    temp_support_dq = temp_support_t75 ** 2 - temp_support_t25 ** 2

    rain_delta = beta_rain * rain_step[None, :]
    rain_delta_support = beta_rain * rain_support[None, :]
    temp_delta = beta_temp * 1.0 + beta_temp_sq * 1.0
    temp_delta_support = beta_temp * temp_support_delta[None, :] + beta_temp_sq * temp_support_dq[None, :]

    payload = {
        "model": spec["name"],
        "fold": bundle["fold"],
        "sample_cap": int(len(take)),
        "rain_step_mm": 10.0,
        "rain_baseline_definition": "city training-period median daily precipitation",
        "temp_baseline_definition": "city training-period mean daily UTCI, evaluated at T = 0 so that a +10 C step gives delta T = 1 and delta Q = 1",
        "rain_10mm_city_equal_percent": interval(100.0 * np.mean(np.expm1(rain_delta), axis=1)),
        "temp_10c_city_equal_percent": interval(100.0 * np.mean(np.expm1(temp_delta), axis=1)),
        "rain_support_city_equal_percent": interval(100.0 * np.mean(np.expm1(rain_delta_support), axis=1)),
        "temp_support_city_equal_percent": interval(100.0 * np.mean(np.expm1(temp_delta_support), axis=1)),
        "rain_support_definition": "city training median to 90th percentile of daily precipitation",
        "temp_support_definition": "city training 25th to 75th percentile of daily UTCI",
        "p_population_rain_negative": float(np.mean(flat_posterior(idata, "beta_rain_population")[take] < 0)),
        "p_population_temp_positive": float(np.mean(flat_posterior(idata, "beta_temp_population")[take] > 0)),
    }

    for name in ("beta_rain_population", "beta_temp_population", "beta_temp_sq_population", "nu"):
        payload[f"{name}_interval"] = interval(flat_posterior(idata, name)[take])
    for name in ("beta_rain_city_sd", "beta_temp_city_sd", "beta_temp_sq_city_sd"):
        if name in idata.posterior:
            payload[f"{name}_interval"] = interval(flat_posterior(idata, name)[take])
    for name in ("beta_rain_moderator", "beta_temp_moderator", "beta_temp_sq_moderator"):
        if name in idata.posterior:
            payload[f"{name}_interval"] = interval(flat_posterior(idata, name)[take])

    if spec["ar1"]:
        payload["phi_city_interval"] = interval(flat_posterior(idata, "phi_city")[take].ravel())
        payload["z_population_interval"] = interval(flat_posterior(idata, "z_population")[take])
        payload["p_phi_positive_city_mean"] = float(
            np.mean(flat_posterior(idata, "phi_city")[take].mean(axis=1) > 0)
        )

    cities = []
    for index, city in enumerate(names):
        valid = beta_temp_sq[:, index] < 0
        peak = temp_mean[index] - 5.0 * beta_temp[:, index] / beta_temp_sq[:, index]
        inside = valid & (peak >= temp_min[index]) & (peak <= temp_max[index])
        cities.append({
            "city": city,
            "rain_10mm_percent": interval(100.0 * np.expm1(rain_delta[:, index])),
            "rain_support_percent": interval(100.0 * np.expm1(rain_delta_support[:, index])),
            "temp_10c_percent": interval(100.0 * np.expm1(temp_delta[:, index])),
            "p_rain_negative": float(np.mean(beta_rain[:, index] < 0)),
            "p_concave_temperature": float(np.mean(valid)),
            "p_interior_temperature_maximum": float(np.mean(inside)),
            "temperature_maximum_c_conditional": (
                interval(peak[inside]) if int(inside.sum()) > 30 else None
            ),
            "observed_utci_min_c": float(temp_min[index]),
            "observed_utci_max_c": float(temp_max[index]),
            "temp_10c_within_observed_support": bool(
                (temp_mean[index] + 10.0) <= temp_max[index]
            ),
        })
    payload["cities"] = cities
    payload["cities_with_interior_maximum_probability_above_0_95"] = int(
        sum(1 for item in cities if item["p_interior_temperature_maximum"] > 0.95)
    )
    payload["cities_with_concave_probability_above_0_95"] = int(
        sum(1 for item in cities if item["p_concave_temperature"] > 0.95)
    )
    payload["cities_with_temp_10c_outside_support"] = int(
        sum(1 for item in cities if not item["temp_10c_within_observed_support"])
    )
    return payload


def effects_table(payload):
    frame = pd.DataFrame(payload["cities"])
    frame["model"] = payload["model"]
    frame["fold"] = payload["fold"]
    return frame
