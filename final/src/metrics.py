import numpy as np
import pandas as pd

WIS_LEVELS = (0.50, 0.80, 0.90, 0.95)
WIS_SOURCE = (
    "Bracher, Ray, Gneiting & Reich (2021), Evaluating epidemic forecasts in an interval "
    "format, PLoS Computational Biology 17(2): e1008618, doi:10.1371/journal.pcbi.1008618"
)


def interval_columns(level):
    tag = f"{int(round(level * 100)):02d}"
    return f"lower_{tag}", f"upper_{tag}"


def weighted_interval_score(y, median, lowers, uppers, alphas):
    y = np.asarray(y, dtype=float)
    score = 0.5 * np.abs(y - np.asarray(median, dtype=float))
    for alpha, low, high in zip(alphas, lowers, uppers):
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
        interval_score = (high - low)
        interval_score = interval_score + (2.0 / alpha) * np.clip(low - y, 0.0, None)
        interval_score = interval_score + (2.0 / alpha) * np.clip(y - high, 0.0, None)
        score = score + (alpha / 2.0) * interval_score
    return score / (len(alphas) + 0.5)


def add_scores(frame, median_col="median_log"):
    alphas = [1.0 - level for level in WIS_LEVELS]
    lowers = [frame[interval_columns(level)[0]].to_numpy() for level in WIS_LEVELS]
    uppers = [frame[interval_columns(level)[1]].to_numpy() for level in WIS_LEVELS]
    result = frame.copy()
    result["wis_log"] = weighted_interval_score(
        result["observed_log"].to_numpy(),
        result[median_col].to_numpy(),
        lowers,
        uppers,
        alphas,
    )
    result["ae_log"] = np.abs(result["observed_log"].to_numpy() - result[median_col].to_numpy())
    result["ae_usage"] = np.abs(
        result["observed_usage"].to_numpy() - result["median_usage"].to_numpy()
    )
    result["se_log"] = (result["observed_log"].to_numpy() - result[median_col].to_numpy()) ** 2
    for level in WIS_LEVELS:
        tag = int(round(level * 100))
        low, high = interval_columns(level)
        result[f"covered_{tag:02d}"] = (
            (result["observed_log"] >= result[low]) & (result["observed_log"] <= result[high])
        )
        result[f"width_{tag:02d}"] = result[high] - result[low]
    return result


def summarize(frame, scale=None):
    summary = {
        "rows": int(len(frame)),
        "wis_log": float(frame["wis_log"].mean()),
        "mean_log_predictive_density": float(frame["log_predictive_density"].mean()),
        "mae_log": float(frame["ae_log"].mean()),
        "rmse_log": float(np.sqrt(frame["se_log"].mean())),
        "mae_usage": float(frame["ae_usage"].mean()),
    }
    for level in WIS_LEVELS:
        tag = int(round(level * 100))
        summary[f"coverage_{tag:02d}"] = float(frame[f"covered_{tag:02d}"].mean())
        summary[f"width_{tag:02d}"] = float(frame[f"width_{tag:02d}"].mean())
    if scale is not None:
        denominator = frame["city"].map(scale)
        if denominator.isna().any() or (denominator <= 0).any():
            raise ValueError("normalisation scale must be positive for every city")
        summary["normalised_mae_log"] = float((frame["ae_log"] / denominator).mean())
    return summary


def city_equal_summary(frame, scale=None):
    per_city = {}
    for city, block in frame.groupby("city", sort=True):
        per_city[city] = summarize(block)
    per_city_frame = pd.DataFrame.from_dict(per_city, orient="index")
    per_city_frame.index.name = "city"
    if scale is not None:
        ratios = frame.assign(ratio=frame["ae_log"] / frame["city"].map(scale))
        per_city_frame["normalised_mae_log"] = ratios.groupby("city", sort=True)["ratio"].mean()

    overall = summarize(frame, scale=scale)
    overall["city_equal_wis_log"] = float(per_city_frame["wis_log"].mean())
    overall["city_equal_mean_log_predictive_density"] = float(
        per_city_frame["mean_log_predictive_density"].mean()
    )
    overall["city_equal_mae_log"] = float(per_city_frame["mae_log"].mean())
    overall["city_equal_mae_usage"] = float(per_city_frame["mae_usage"].mean())
    overall["city_equal_coverage_90"] = float(per_city_frame["coverage_90"].mean())
    worst = per_city_frame["wis_log"].idxmax()
    overall["worst_city_wis"] = worst
    overall["worst_city_wis_value"] = float(per_city_frame.loc[worst, "wis_log"])
    overall["worst_city_coverage_90"] = float(per_city_frame.loc[worst, "coverage_90"])
    return overall, per_city_frame


def seasonal_naive_scale(train_frame, lag=7):
    scales = {}
    for city, block in train_frame.groupby("city", sort=True):
        block = block.sort_values("date")
        response = block["usage_log1p"].to_numpy()
        if len(response) <= lag:
            raise ValueError(f"not enough training rows for the seasonal naive scale in {city}")
        scales[city] = float(np.mean(np.abs(response[lag:] - response[:-lag])))
    if any(value <= 0 for value in scales.values()):
        raise ValueError("seasonal naive scale must be positive for every city")
    return scales
