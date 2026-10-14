import numpy as np
import pandas as pd

CITY = "city"
DATE = "date"
RESPONSE = "usage_log1p"

PREDICTION_ROSTER_START = pd.Timestamp("2017-01-02")
PREDICTION_ROSTER_END = pd.Timestamp("2017-12-31")
PREDICTION_TEST_START = pd.Timestamp("2017-07-17")


def load_daily(data_path):
    frame = pd.read_csv(data_path, parse_dates=[DATE])
    frame = frame.sort_values([CITY, DATE]).reset_index(drop=True)
    if frame.duplicated([CITY, DATE]).any():
        raise ValueError("duplicate city-date keys in frozen input")
    if frame.isna().any().any():
        raise ValueError("frozen input contains missing values")
    expected_sin = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
    expected_cos = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)
    if not np.allclose(frame["season_sin"], expected_sin, atol=1e-12):
        raise ValueError("season_sin column disagrees with day_of_year")
    if not np.allclose(frame["season_cos"], expected_cos, atol=1e-12):
        raise ValueError("season_cos column disagrees with day_of_year")
    return frame


def roster_primary(frame):
    names = sorted(frame.loc[frame["continuous_system"] == 1, CITY].unique())
    if len(names) != 25:
        raise ValueError(f"primary roster expected 25 cities, found {len(names)}")
    return names


def roster_prediction(frame):
    span = frame.groupby(CITY)[DATE].agg(["min", "max"])
    continuous = frame.groupby(CITY)["continuous_system"].first()
    keep = span.index[
        (span["min"] <= PREDICTION_ROSTER_START)
        & (span["max"] >= PREDICTION_ROSTER_END)
        & (continuous.reindex(span.index) == 1)
    ]
    names = sorted(keep.tolist())
    if len(names) != 23:
        raise ValueError(f"prediction roster expected 23 cities, found {len(names)}")
    return names


def roster_full(frame):
    return sorted(frame[CITY].unique())


def standardize(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if not sd > 0:
        raise ValueError("standardisation scale must be positive")
    return (values - mean) / sd, mean, sd


def fit_preprocessing(train, city_names):
    train = train.loc[train[CITY].isin(city_names)]
    if train.empty:
        raise ValueError("no training rows for the requested roster")
    city_temp_mean = train.groupby(CITY)["utci_mean_c"].mean().reindex(city_names)
    city_precip_mean = train.groupby(CITY)["precipitation_log1p"].mean().reindex(city_names)
    if city_temp_mean.isna().any() or city_precip_mean.isna().any():
        raise ValueError("a roster city has no training rows")

    temp = (train["utci_mean_c"] - train[CITY].map(city_temp_mean)) / 10.0
    temp_sq_center = pd.Series(temp.to_numpy() ** 2, index=train.index).groupby(train[CITY]).mean()
    rain_centered = train["precipitation_log1p"] - train[CITY].map(city_precip_mean)
    rain_sd = float(rain_centered.std(ddof=1))
    if not rain_sd > 0:
        raise ValueError("pooled within-city rain scale must be positive")

    thermal_raw = city_temp_mean.to_numpy()
    rain_raw = np.log1p(train.groupby(CITY)["precipitation_mm"].mean().reindex(city_names).to_numpy())
    thermal_z, thermal_mean, thermal_sd = standardize(thermal_raw)
    rain_baseline_z, rain_baseline_mean, rain_baseline_sd = standardize(rain_raw)

    day_number = (train[DATE] - train[DATE].min()).dt.days
    trend_center = pd.Series(day_number.to_numpy(), index=train.index).groupby(train[CITY]).mean()

    return {
        "city_names": list(city_names),
        "city_temp_mean_c": city_temp_mean.to_dict(),
        "city_precip_log1p_mean": city_precip_mean.to_dict(),
        "temp_sq_center": temp_sq_center.to_dict(),
        "trend_center_day": trend_center.to_dict(),
        "day_origin": str(train[DATE].min().date()),
        "rain_pooled_sd": rain_sd,
        "thermal_baseline_z": dict(zip(city_names, thermal_z.tolist())),
        "rain_baseline_z": dict(zip(city_names, rain_baseline_z.tolist())),
        "thermal_baseline_mean": thermal_mean,
        "thermal_baseline_sd": thermal_sd,
        "rain_baseline_mean": rain_baseline_mean,
        "rain_baseline_sd": rain_baseline_sd,
        "train_rows": int(len(train)),
        "train_date_min": str(train[DATE].min().date()),
        "train_date_max": str(train[DATE].max().date()),
    }


def apply_preprocessing(frame, prep):
    city_names = prep["city_names"]
    frame = frame.loc[frame[CITY].isin(city_names)].copy()
    if frame.empty:
        raise ValueError("no rows to transform")
    lookup = {name: index for index, name in enumerate(city_names)}
    city_index = frame[CITY].map(lookup).to_numpy()
    if np.isnan(city_index).any():
        raise ValueError("unknown city encountered during transform")
    city_index = city_index.astype(int)

    temp = (frame["utci_mean_c"] - frame[CITY].map(prep["city_temp_mean_c"])) / 10.0
    temp_sq = temp.to_numpy() ** 2 - frame[CITY].map(prep["temp_sq_center"]).to_numpy()
    rain = (
        (frame["precipitation_log1p"] - frame[CITY].map(prep["city_precip_log1p_mean"]))
        / prep["rain_pooled_sd"]
    ).to_numpy()

    day_origin = pd.Timestamp(prep["day_origin"])
    day_number = (frame[DATE] - day_origin).dt.days
    trend = (day_number - frame[CITY].map(prep["trend_center_day"])) / 100.0

    arrays = {
        "city_index": city_index,
        "temp": temp.to_numpy(dtype=float),
        "temp_sq": np.asarray(temp_sq, dtype=float),
        "rain": np.asarray(rain, dtype=float),
        "weekday": frame["weekday"].to_numpy(dtype=int),
        "weekend": frame["weekend"].to_numpy(dtype=float),
        "season_sin": frame["season_sin"].to_numpy(dtype=float),
        "season_cos": frame["season_cos"].to_numpy(dtype=float),
        "trend": trend.to_numpy(dtype=float),
        "response": frame[RESPONSE].to_numpy(dtype=float),
        "thermal_baseline": np.array([prep["thermal_baseline_z"][name] for name in city_names]),
        "rain_baseline": np.array([prep["rain_baseline_z"][name] for name in city_names]),
    }
    arrays["restart"] = segment_restart(city_index, frame[DATE].to_numpy())
    return frame.reset_index(drop=True), arrays


def segment_restart(city_index, dates):
    dates = pd.to_datetime(pd.Series(dates))
    restart = np.ones(len(city_index), dtype=float)
    for position in range(1, len(city_index)):
        if city_index[position] != city_index[position - 1]:
            continue
        gap = (dates.iloc[position] - dates.iloc[position - 1]).days
        restart[position] = 0.0 if gap == 1 else 1.0
    return restart


def last_training_row(arrays):
    city_index = arrays["city_index"]
    last = np.full(len(arrays["thermal_baseline"]), -1, dtype=int)
    for position in range(len(city_index)):
        last[city_index[position]] = position
    if (last < 0).any():
        raise ValueError("a roster city has no training observation")
    return last


def build_split(frame, city_names, cutoff, test_start, test_end):
    roster = frame.loc[frame[CITY].isin(city_names)]
    train = roster.loc[roster[DATE] <= cutoff]
    test = roster.loc[(roster[DATE] >= test_start) & (roster[DATE] <= test_end)]
    if train.empty or test.empty:
        raise ValueError("empty train or test partition")
    prep = fit_preprocessing(train, city_names)
    train_frame, train_arrays = apply_preprocessing(train, prep)
    test_frame, test_arrays = apply_preprocessing(test, prep)
    return {
        "prep": prep,
        "train_frame": train_frame,
        "train_arrays": train_arrays,
        "test_frame": test_frame,
        "test_arrays": test_arrays,
    }
