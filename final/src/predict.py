import numpy as np
import pandas as pd
from scipy import stats

from data import CITY, DATE, last_training_row
from metrics import WIS_LEVELS, interval_columns
from model import flat_posterior, linear_predictor, select_samples

PREDICTIVE_SAMPLE_CAP = 400
PATH_COUNT = 200


def test_restart(test_frame, train_frame, city_names):
    last_train = train_frame.groupby(CITY)[DATE].max().reindex(city_names)
    restart = np.ones(len(test_frame), dtype=float)
    previous_date = None
    previous_city = None
    for position, (city, date) in enumerate(zip(test_frame[CITY], test_frame[DATE])):
        if city != previous_city:
            gap = (date - last_train[city]).days
        else:
            gap = (date - previous_date).days
        restart[position] = 0.0 if gap == 1 else 1.0
        previous_date = date
        previous_city = city
    return restart


def _quantile_frame(city, dates, observed_log, observed_usage, samples, density):
    frame = pd.DataFrame({
        "city": city,
        "date": dates,
        "observed_log": observed_log,
        "observed_usage": observed_usage,
        "median_log": np.median(samples, axis=0),
        "log_predictive_density": density,
    })
    for level in WIS_LEVELS:
        low, high = interval_columns(level)
        tail = (1.0 - level) / 2.0
        frame[low] = np.quantile(samples, tail, axis=0)
        frame[high] = np.quantile(samples, 1.0 - tail, axis=0)
    frame["median_usage"] = np.expm1(frame["median_log"])
    return frame


def _mixture_density(values, nu, mu, sigma):
    components = stats.t.logpdf(values[None, :], nu[:, None], mu, sigma)
    maximum = components.max(axis=0)
    return maximum + np.log(np.mean(np.exp(components - maximum[None, :]), axis=0))


def _mixture_density_3d(values, nu, mu, sigma):
    components = stats.t.logpdf(values[None, None, :], nu[:, None, None], mu, sigma)
    maximum = components.max(axis=(0, 1))
    return maximum + np.log(
        np.mean(np.exp(components - maximum[None, None, :]), axis=(0, 1))
    )


def _ar_paths(rng, mu_city, phi, sigma, sigma_init, nu, error_last, restart, paths, observed_log):
    sample_count, horizon = mu_city.shape
    error = np.empty((sample_count, paths, horizon))
    conditional_mu = np.empty((sample_count, paths, horizon))
    conditional_sigma = np.empty((sample_count, paths, horizon))
    previous_state = np.repeat(error_last[:, None], paths, axis=1)
    for step in range(horizon):
        offset = phi[:, None] * previous_state
        offset = np.where(restart[step] == 1.0, 0.0, offset)
        conditional_mu[:, :, step] = mu_city[:, step][:, None] + offset
        conditional_sigma[:, :, step] = np.where(
            restart[step] == 1.0, sigma_init[:, None], sigma[:, None]
        )
        innovation = rng.standard_t(nu[:, None], size=(sample_count, paths)) * sigma[:, None]
        propagated = phi[:, None] * previous_state + innovation
        initial = rng.standard_t(nu[:, None], size=(sample_count, paths)) * sigma_init[:, None]
        error[:, :, step] = np.where(restart[step] == 1.0, initial, propagated)
        previous_state = error[:, :, step]
    samples = (mu_city[:, None, :] + error).reshape(-1, horizon)
    density = _mixture_density_3d(observed_log, nu, conditional_mu, conditional_sigma)
    return samples, density


def block_forecast(idata, spec, bundle, sample_cap=PREDICTIVE_SAMPLE_CAP, paths=PATH_COUNT,
                   seed=27, city_names=None):
    names = city_names if city_names is not None else bundle["prep"]["city_names"]
    take = select_samples(idata, sample_cap, seed)
    test_arrays = bundle["test_arrays"]
    test_frame = bundle["test_frame"]
    train_frame = bundle["train_frame"]
    train_arrays = bundle["train_arrays"]
    mu_test = linear_predictor(idata, spec, test_arrays, take)
    sigma = flat_posterior(idata, "sigma_city")[take]
    nu = flat_posterior(idata, "nu")[take]
    rng = np.random.default_rng(seed)

    phi = None
    sigma_init = None
    error_last = None
    if spec["ar1"]:
        phi = flat_posterior(idata, "phi_city")[take]
        sigma_init = flat_posterior(idata, "sigma_init")[take]
        last_rows = last_training_row(train_arrays)
        mu_train_last = linear_predictor(idata, spec, train_arrays, take)[:, last_rows]
        observed_last = train_arrays["response"][last_rows]
        error_last = observed_last[None, :] - mu_train_last

    restart = test_restart(test_frame, train_frame, names)
    frames = []
    stability = 0.0
    for index, city in enumerate(names):
        rows = np.flatnonzero(test_arrays["city_index"] == index)
        if len(rows) == 0:
            raise ValueError(f"no test rows for {city}")
        mu_city = mu_test[:, rows]
        sigma_city = sigma[:, index]
        restart_city = restart[rows]
        observed_log = test_arrays["response"][rows]

        if spec["ar1"]:
            samples, density = _ar_paths(
                rng, mu_city, phi[:, index], sigma_city, sigma_init[:, index],
                nu, error_last[:, index], restart_city, paths, observed_log,
            )
            reduced_samples, reduced_density = _ar_paths(
                rng, mu_city, phi[:, index], sigma_city, sigma_init[:, index],
                nu, error_last[:, index], restart_city, max(2, paths // 4), observed_log,
            )
            stability = max(stability, float(np.max(np.abs(density - reduced_density))))
            del reduced_samples
        else:
            draws = rng.standard_t(nu[:, None, None], size=(len(take), paths, len(rows)))
            samples = (mu_city[:, None, :] + sigma_city[:, None, None] * draws).reshape(
                -1, len(rows)
            )
            density = _mixture_density(observed_log, nu, mu_city, sigma_city[:, None])

        frame = _quantile_frame(
            city, test_frame.loc[rows, DATE].to_numpy(), observed_log,
            test_frame.loc[rows, "usage"].to_numpy(), samples, density,
        )
        frame["horizon"] = np.arange(1, len(rows) + 1)
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result["task"] = "block"
    result["model"] = spec["name"]
    result["fold"] = bundle["fold"]
    return result, {"density_stability_max_abs": float(stability)}


def one_step_forecast(idata, spec, bundle, sample_cap=PREDICTIVE_SAMPLE_CAP, paths=PATH_COUNT,
                      seed=27, city_names=None):
    names = city_names if city_names is not None else bundle["prep"]["city_names"]
    take = select_samples(idata, sample_cap, seed)
    test_arrays = bundle["test_arrays"]
    test_frame = bundle["test_frame"]
    train_frame = bundle["train_frame"]
    train_arrays = bundle["train_arrays"]
    mu_test = linear_predictor(idata, spec, test_arrays, take)
    sigma = flat_posterior(idata, "sigma_city")[take]
    nu = flat_posterior(idata, "nu")[take]
    rng = np.random.default_rng(seed + 1)

    phi = None
    sigma_init = None
    if spec["ar1"]:
        phi = flat_posterior(idata, "phi_city")[take]
        sigma_init = flat_posterior(idata, "sigma_init")[take]

    restart = test_restart(test_frame, train_frame, names)
    last_rows = last_training_row(train_arrays)
    mu_train_last = linear_predictor(idata, spec, train_arrays, take)[:, last_rows]
    observed_last = train_arrays["response"][last_rows]

    frames = []
    for index, city in enumerate(names):
        rows = np.flatnonzero(test_arrays["city_index"] == index)
        mu_city = mu_test[:, rows]
        sigma_city = sigma[:, index]
        restart_city = restart[rows]
        observed_log = test_arrays["response"][rows]

        if spec["ar1"]:
            previous_error = observed_last[index] - mu_train_last[:, index]
            conditional_mu = np.empty_like(mu_city)
            conditional_sigma = np.empty_like(mu_city)
            for step in range(len(rows)):
                if restart_city[step] == 1.0:
                    conditional_mu[:, step] = mu_city[:, step]
                    conditional_sigma[:, step] = sigma_init[:, index]
                else:
                    conditional_mu[:, step] = mu_city[:, step] + phi[:, index] * previous_error
                    conditional_sigma[:, step] = sigma_city
                previous_error = observed_log[step] - mu_city[:, step]
        else:
            conditional_mu = mu_city
            conditional_sigma = np.repeat(sigma_city[:, None], len(rows), axis=1)

        density = _mixture_density(observed_log, nu, conditional_mu, conditional_sigma)
        draws = rng.standard_t(nu[:, None, None], size=(len(take), paths, len(rows)))
        samples = (
            conditional_mu[:, None, :] + conditional_sigma[:, None, :] * draws
        ).reshape(-1, len(rows))
        frame = _quantile_frame(
            city, test_frame.loc[rows, DATE].to_numpy(), observed_log,
            test_frame.loc[rows, "usage"].to_numpy(), samples, density,
        )
        frame["horizon"] = np.arange(1, len(rows) + 1)
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result["task"] = "one_step"
    result["model"] = spec["name"]
    result["fold"] = bundle["fold"]
    return result, {"density_stability_max_abs": 0.0}
