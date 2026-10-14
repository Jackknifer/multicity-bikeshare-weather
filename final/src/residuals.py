import numpy as np
import pandas as pd

from data import CITY, DATE
from model import flat_posterior, linear_predictor, select_samples

RESIDUAL_SAMPLE_CAP = 400
MAX_LAG = 7


def residual_table(idata, spec, bundle, seed=27, sample_cap=RESIDUAL_SAMPLE_CAP):
    take = select_samples(idata, sample_cap, seed)
    arrays = bundle["train_arrays"]
    mu = linear_predictor(idata, spec, arrays, take)
    observed = arrays["response"]
    error = observed[None, :] - mu

    if spec["ar1"]:
        phi = flat_posterior(idata, "phi_city")[take]
        previous = np.empty_like(error)
        previous[:, 0] = 0.0
        previous[:, 1:] = error[:, :-1]
        rho = phi[:, arrays["city_index"]]
        restart = arrays["restart"][None, :]
        innovation = np.where(restart == 1.0, error, error - rho * previous)
        residual = np.median(innovation, axis=0)
        residual_kind = "innovation"
    else:
        residual = np.median(error, axis=0)
        residual_kind = "residual"

    frame = bundle["train_frame"][[CITY, DATE]].copy()
    frame["residual"] = residual
    frame["restart"] = arrays["restart"]
    return frame, residual_kind


def autocorrelation(frame, max_lag=MAX_LAG):
    records = []
    for city, block in frame.groupby(CITY, sort=True):
        block = block.sort_values(DATE)
        values = block["residual"].to_numpy()
        dates = block[DATE].to_numpy()
        gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
        contiguous = gaps == 1
        for lag in range(1, max_lag + 1):
            pairs = contiguous[:len(contiguous) - lag + 1] if lag > 1 else contiguous
            left = values[:len(values) - lag][pairs]
            right = values[lag:][pairs]
            if len(left) < 30:
                raise ValueError(f"too few contiguous pairs for {city} at lag {lag}")
            records.append({
                "city": city,
                "lag": lag,
                "autocorrelation": float(np.corrcoef(left, right)[0, 1]),
                "pairs": int(len(left)),
            })
    return pd.DataFrame(records)


def posterior_predictive_check(idata, spec, bundle, seed=27, sample_cap=RESIDUAL_SAMPLE_CAP,
                               paths=200):
    take = select_samples(idata, sample_cap, seed)
    arrays = bundle["train_arrays"]
    mu = linear_predictor(idata, spec, arrays, take)
    sigma = flat_posterior(idata, "sigma_city")[take][:, arrays["city_index"]]
    nu = flat_posterior(idata, "nu")[take]
    observed = arrays["response"]

    if spec["ar1"]:
        phi = flat_posterior(idata, "phi_city")[take][:, arrays["city_index"]]
        previous = np.empty_like(mu)
        previous[:, 0] = 0.0
        previous[:, 1:] = observed[None, :-1] - mu[:, :-1]
        restart = arrays["restart"][None, :]
        conditional_mu = np.where(restart == 1.0, mu, mu + phi * previous)
        sigma_init = flat_posterior(idata, "sigma_init")[take][:, arrays["city_index"]]
        conditional_sigma = np.where(restart == 1.0, sigma_init, sigma)
    else:
        conditional_mu = mu
        conditional_sigma = sigma

    rng = np.random.default_rng(seed)
    draws = rng.standard_t(nu[:, None, None], size=(len(take), paths, len(observed)))
    simulated = conditional_mu[:, None, :] + conditional_sigma[:, None, :] * draws
    lower = np.quantile(simulated, 0.05, axis=(0, 1))
    upper = np.quantile(simulated, 0.95, axis=(0, 1))

    frame = bundle["train_frame"][[CITY, DATE]].copy()
    frame["observed_log"] = observed
    frame["location_median_log"] = np.median(conditional_mu, axis=0)
    frame["lower90_log"] = lower
    frame["upper90_log"] = upper
    frame["covered90"] = (observed >= lower) & (observed <= upper)
    frame["residual_log"] = observed - frame["location_median_log"].to_numpy()
    return frame, float(np.mean(simulated < 0))
