import numpy as np
import pandas as pd
from scipy import stats

from data import CITY, DATE
from metrics import WIS_LEVELS, interval_columns

OLS_FEATURE_NAMES = (
    "intercept", "temp", "temp_sq", "rain", "weekday_1", "weekday_2", "weekday_3",
    "weekday_4", "weekday_5", "weekday_6", "trend", "season_sin", "season_cos",
)


def _design(arrays):
    columns = [
        np.ones(len(arrays["response"])),
        arrays["temp"],
        arrays["temp_sq"],
        arrays["rain"],
    ]
    for day in range(1, 7):
        columns.append((arrays["weekday"] == day).astype(float))
    columns.append(arrays["trend"])
    columns.append(arrays["season_sin"])
    columns.append(arrays["season_cos"])
    return np.column_stack(columns)


def _finalize(city, dates, observed_log, observed_usage, point, density, distribution):
    frame = pd.DataFrame({
        "city": city,
        "date": dates,
        "observed_log": observed_log,
        "observed_usage": observed_usage,
        "median_log": point,
        "log_predictive_density": density,
    })
    for level in WIS_LEVELS:
        low, high = interval_columns(level)
        tail = (1.0 - level) / 2.0
        frame[low] = distribution.ppf(tail)
        frame[high] = distribution.ppf(1.0 - tail)
    frame["median_usage"] = np.expm1(frame["median_log"])
    return frame


def city_constant(bundle):
    frames = []
    for index, city in enumerate(bundle["prep"]["city_names"]):
        train_rows = bundle["train_arrays"]["city_index"] == index
        test_rows = bundle["test_arrays"]["city_index"] == index
        point_value = float(np.median(bundle["train_arrays"]["response"][train_rows]))
        sigma = float(np.std(bundle["train_arrays"]["response"][train_rows], ddof=1))
        observed_log = bundle["test_arrays"]["response"][test_rows]
        point = np.full(len(observed_log), point_value)
        distribution = stats.norm(point, sigma)
        for task in ("block", "one_step"):
            frame = _finalize(
                city, bundle["test_frame"].loc[test_rows, DATE].to_numpy(), observed_log,
                bundle["test_frame"].loc[test_rows, "usage"].to_numpy(), point,
                distribution.logpdf(observed_log), distribution,
            )
            frame["horizon"] = np.arange(1, len(observed_log) + 1)
            frame["task"] = task
            frame["model"] = "city_constant"
            frame["fold"] = bundle["fold"]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def seasonal_naive(bundle, lag=7):
    frames = []
    for index, city in enumerate(bundle["prep"]["city_names"]):
        train_rows = bundle["train_arrays"]["city_index"] == index
        test_rows = bundle["test_arrays"]["city_index"] == index
        train_response = bundle["train_arrays"]["response"][train_rows]
        observed_log = bundle["test_arrays"]["response"][test_rows]
        horizon = len(observed_log)
        differences = train_response[lag:] - train_response[:-lag]
        sigma = float(np.std(differences, ddof=1))

        history = list(train_response[-lag:])
        recursive = []
        for step in range(horizon):
            recursive.append(history[step])
            history.append(recursive[-1])
        block_point = np.array(recursive)

        previous = np.concatenate([train_response[-lag:], observed_log])
        one_step_point = previous[:horizon]

        for task, point in (("block", block_point), ("one_step", one_step_point)):
            distribution = stats.norm(point, sigma)
            frame = _finalize(
                city, bundle["test_frame"].loc[test_rows, DATE].to_numpy(), observed_log,
                bundle["test_frame"].loc[test_rows, "usage"].to_numpy(), point,
                distribution.logpdf(observed_log), distribution,
            )
            frame["horizon"] = np.arange(1, horizon + 1)
            frame["task"] = task
            frame["model"] = "seasonal_naive_7"
            frame["fold"] = bundle["fold"]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def city_ols(bundle):
    frames = []
    for index, city in enumerate(bundle["prep"]["city_names"]):
        train_rows = bundle["train_arrays"]["city_index"] == index
        test_rows = bundle["test_arrays"]["city_index"] == index
        train_design = _design(bundle["train_arrays"])[train_rows]
        test_design = _design(bundle["test_arrays"])[test_rows]
        train_response = bundle["train_arrays"]["response"][train_rows]
        observed_log = bundle["test_arrays"]["response"][test_rows]
        coefficients = np.linalg.lstsq(train_design, train_response, rcond=None)[0]
        residuals = train_response - train_design @ coefficients
        degrees = len(train_response) - train_design.shape[1]
        variance = float(residuals @ residuals) / degrees
        leverage = np.sum((test_design @ np.linalg.pinv(train_design.T @ train_design)) * test_design, axis=1)
        scale = np.sqrt(variance * (1.0 + leverage))
        point = test_design @ coefficients
        distribution = stats.t(degrees, point, scale)
        for task in ("block", "one_step"):
            frame = _finalize(
                city, bundle["test_frame"].loc[test_rows, DATE].to_numpy(), observed_log,
                bundle["test_frame"].loc[test_rows, "usage"].to_numpy(), point,
                distribution.logpdf(observed_log), distribution,
            )
            frame["horizon"] = np.arange(1, len(observed_log) + 1)
            frame["task"] = task
            frame["model"] = "city_ols"
            frame["fold"] = bundle["fold"]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def city_ar_ols(bundle):
    frames = []
    for index, city in enumerate(bundle["prep"]["city_names"]):
        train_rows = bundle["train_arrays"]["city_index"] == index
        test_rows = bundle["test_arrays"]["city_index"] == index
        train_design = _design(bundle["train_arrays"])[train_rows]
        test_design = _design(bundle["test_arrays"])[test_rows]
        train_response = bundle["train_arrays"]["response"][train_rows]
        observed_log = bundle["test_arrays"]["response"][test_rows]
        coefficients = np.linalg.lstsq(train_design, train_response, rcond=None)[0]
        residuals = train_response - train_design @ coefficients
        rho = float(np.sum(residuals[1:] * residuals[:-1]) / np.sum(residuals[:-1] ** 2))
        innovations = residuals[1:] - rho * residuals[:-1]
        sigma = float(np.std(innovations, ddof=1))
        degrees = len(innovations) - 1

        previous_error = residuals[-1]
        block_point = []
        for step in range(len(observed_log)):
            value = test_design[step] @ coefficients + rho * previous_error
            block_point.append(value)
            previous_error = block_point[-1] - test_design[step] @ coefficients
        block_point = np.array(block_point)

        one_step_point = np.empty(len(observed_log))
        previous_error = residuals[-1]
        for step in range(len(observed_log)):
            one_step_point[step] = test_design[step] @ coefficients + rho * previous_error
            previous_error = observed_log[step] - test_design[step] @ coefficients

        for task, point in (("block", block_point), ("one_step", one_step_point)):
            distribution = stats.t(degrees, point, sigma)
            frame = _finalize(
                city, bundle["test_frame"].loc[test_rows, DATE].to_numpy(), observed_log,
                bundle["test_frame"].loc[test_rows, "usage"].to_numpy(), point,
                distribution.logpdf(observed_log), distribution,
            )
            frame["horizon"] = np.arange(1, len(observed_log) + 1)
            frame["task"] = task
            frame["model"] = "city_ar_ols"
            frame["fold"] = bundle["fold"]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


BASELINES = {
    "city_constant": city_constant,
    "seasonal_naive_7": seasonal_naive,
    "city_ols": city_ols,
    "city_ar_ols": city_ar_ols,
}
