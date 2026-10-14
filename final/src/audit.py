import numpy as np
import pandas as pd

from data import (
    CITY,
    DATE,
    PREDICTION_ROSTER_END,
    PREDICTION_ROSTER_START,
    load_daily,
    roster_full,
    roster_prediction,
    roster_primary,
)


def city_audit(frame):
    records = []
    primary = set(roster_primary(frame))
    prediction = set(roster_prediction(frame))
    for city, block in frame.groupby(CITY, sort=True):
        block = block.sort_values(DATE)
        span = pd.date_range(block[DATE].min(), block[DATE].max(), freq="D")
        observed = pd.DatetimeIndex(block[DATE])
        missing = span.difference(observed)
        gaps = np.diff(observed.to_numpy()).astype("timedelta64[D]").astype(int)
        records.append({
            "city": city,
            "country": block["country"].iloc[0],
            "climate_code": block["climate_code"].iloc[0],
            "first_date": block[DATE].min().date().isoformat(),
            "last_date": block[DATE].max().date().isoformat(),
            "rows": int(len(block)),
            "span_days": int(len(span)),
            "missing_days_in_span": int(len(missing)),
            "missing_dates": ";".join(item.date().isoformat() for item in missing),
            "max_calendar_gap_days": int(gaps.max()) if len(gaps) else 0,
            "zero_usage_days": int((block["usage"] == 0).sum()),
            "zero_usage_share": float((block["usage"] == 0).mean()),
            "active_day_share": float(block["active_day_share"].iloc[0]),
            "continuous_system": int(block["continuous_system"].iloc[0]),
            "usage_min": float(block["usage"].min()),
            "usage_median": float(block["usage"].median()),
            "usage_mean": float(block["usage"].mean()),
            "usage_max": float(block["usage"].max()),
            "usage_q25": float(block["usage"].quantile(0.25)),
            "usage_q75": float(block["usage"].quantile(0.75)),
            "utci_mean_c_min": float(block["utci_mean_c"].min()),
            "utci_mean_c_median": float(block["utci_mean_c"].median()),
            "utci_mean_c_max": float(block["utci_mean_c"].max()),
            "precipitation_mm_mean": float(block["precipitation_mm"].mean()),
            "precipitation_mm_q90": float(block["precipitation_mm"].quantile(0.90)),
            "precipitation_mm_max": float(block["precipitation_mm"].max()),
            "wet_day_share": float((block["precipitation_mm"] > 0).mean()),
            "in_primary_25": int(city in primary),
            "in_prediction_23": int(city in prediction),
        })
    audit = pd.DataFrame(records)
    return audit


def integrity_report(frame):
    span = frame.groupby(CITY)[DATE].agg(["min", "max"])
    return {
        "rows": int(len(frame)),
        "cities": int(frame[CITY].nunique()),
        "duplicate_city_dates": int(frame.duplicated([CITY, DATE]).sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "negative_usage_rows": int((frame["usage"] < 0).sum()),
        "negative_precipitation_rows": int((frame["precipitation_mm"] < 0).sum()),
        "zero_usage_rows": int((frame["usage"] == 0).sum()),
        "fractional_usage_rows": int((frame["usage"] % 1 != 0).sum()),
        "cities_starting_before_2017": sorted(span.index[span["min"] < pd.Timestamp("2017-01-01")].tolist()),
        "cities_ending_before_2018": sorted(span.index[span["max"] < PREDICTION_ROSTER_END].tolist()),
        "prediction_roster_start": PREDICTION_ROSTER_START.date().isoformat(),
        "prediction_roster_end": PREDICTION_ROSTER_END.date().isoformat(),
        "primary_roster": roster_primary(frame),
        "prediction_roster": roster_prediction(frame),
        "full_roster": roster_full(frame),
    }


def roster_comparison(frame):
    primary = set(roster_primary(frame))
    prediction = set(roster_prediction(frame))
    return {
        "primary_not_prediction": sorted(primary - prediction),
        "prediction_not_primary": sorted(prediction - primary),
        "prediction_rows": int(frame.loc[frame[CITY].isin(prediction)].shape[0]),
        "primary_rows": int(frame.loc[frame[CITY].isin(primary)].shape[0]),
    }
