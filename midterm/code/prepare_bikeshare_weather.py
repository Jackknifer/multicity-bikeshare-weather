from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "raw" / "bikeshare-weather-40cities"
ARCHIVE_PATH = ROOT / "data" / "raw" / "bikeshare_weather_40cities.zip"
METADATA_PATH = SOURCE_DIR / "bs" / "bs-ll.csv"
STOCK_PATH = SOURCE_DIR / "bs" / "stock-data.csv"
UTCI_PATH = SOURCE_DIR / "utci" / "f_utci.csv"
RAIN_PATH = SOURCE_DIR / "utci" / "f_rain.csv"
OUTPUT_PATH = ROOT / "data" / "clean" / "bikeshare_weather_daily.csv"
AUDIT_PATH = ROOT / "results" / "bikeshare_weather_data_audit.json"
SELECTED_ROWS = list(range(6, 34)) + [35]
EXPECTED_EXTRA_STOCK_CITIES = {"stockholm"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def duplicate_key_records(stock):
    duplicate_rows = stock[stock.duplicated(["city_key", "source_date", "source_hour"], keep=False)]
    records = []
    for key, group in duplicate_rows.groupby(["city_key", "source_date", "source_hour"], sort=True):
        records.append(
            {
                "city_key": key[0],
                "source_date": str(key[1]),
                "source_hour": int(key[2]),
                "row_count": int(len(group)),
                "usage_values": [float(value) for value in group["usage"]],
                "identical_rows": bool(group["usage"].nunique(dropna=False) == 1),
            }
        )
    return records


def build_city_hourly(city_row, stock, utci, rain):
    city = city_row["City"]
    city_key = city.lower()
    city_stock = stock.loc[stock["city_key"] == city_key].copy()
    require(len(city_stock) == 8760, f"{city} must have 8760 stock rows")

    source_timestamp = pd.to_datetime(
        city_stock["source_date"].astype(str)
        + " "
        + city_stock["source_hour"].astype(str).str.zfill(2),
        format="%Y-%m-%d %H",
    )
    start_date = source_timestamp.min().normalize()
    end_date = source_timestamp.max().normalize()
    utc_time = pd.date_range(start=start_date, periods=8760, freq="h", tz="UTC")
    local_time = utc_time.tz_convert(city_row["TZ"])

    city_utci = pd.to_numeric(utci.loc[city], errors="coerce").to_numpy(dtype=float)
    city_rain = pd.to_numeric(rain.loc[city], errors="coerce").to_numpy(dtype=float) * 1000.0
    require(len(city_utci) == 8760, f"{city} must have 8760 UTCI values")
    require(len(city_rain) == 8760, f"{city} must have 8760 rain values")
    require(np.isfinite(city_rain).all(), f"{city} rain contains missing values")
    require((city_rain >= 0).all(), f"{city} rain contains negative values")

    hourly = pd.DataFrame(
        {
            "city": city,
            "city_key": city_key,
            "country": city_row["Country"],
            "climate_code": city_row["Koppen"],
            "latitude": float(city_row["Latitude"]),
            "longitude": float(city_row["Longitude"]),
            "timezone": city_row["TZ"],
            "source_date": city_stock["source_date"].astype(str).to_numpy(),
            "source_hour": city_stock["source_hour"].astype(int).to_numpy(),
            "source_row": np.arange(8760),
            "utc_time": utc_time,
            "local_time": local_time,
            "local_date": local_time.date,
            "usage": city_stock["usage"].astype(float).to_numpy(),
            "utci_c": city_utci,
            "precipitation_mm": city_rain,
        }
    )
    hourly["nominal_date"] = pd.to_datetime(hourly["local_date"])
    hourly["inside_nominal_period"] = hourly["nominal_date"].between(start_date, end_date)
    return hourly


def aggregate_daily(hourly):
    daily = (
        hourly.groupby(
            [
                "city",
                "city_key",
                "country",
                "climate_code",
                "latitude",
                "longitude",
                "timezone",
                "local_date",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            usage=("usage", "sum"),
            utci_mean_c=("utci_c", "mean"),
            utci_min_c=("utci_c", "min"),
            utci_max_c=("utci_c", "max"),
            precipitation_mm=("precipitation_mm", "sum"),
            observed_hours=("source_row", "size"),
            utci_observed_hours=("utci_c", "count"),
            rain_observed_hours=("precipitation_mm", "count"),
            inside_nominal_period=("inside_nominal_period", "all"),
        )
    )
    daily["date"] = pd.to_datetime(daily.pop("local_date"))
    daily["weekday"] = daily["date"].dt.dayofweek
    daily["weekend"] = daily["weekday"].isin([5, 6]).astype(int)
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["season_sin"] = np.sin(2.0 * np.pi * daily["day_of_year"] / 365.25)
    daily["season_cos"] = np.cos(2.0 * np.pi * daily["day_of_year"] / 365.25)
    daily["analysis_ready"] = (
        daily["inside_nominal_period"]
        & daily["observed_hours"].between(23, 25)
        & (daily["utci_observed_hours"] >= 20)
        & (daily["rain_observed_hours"] == daily["observed_hours"])
    )
    return daily


def city_summary(daily):
    records = []
    for city, group in daily.groupby("city", sort=True):
        ready = group.loc[group["analysis_ready"]]
        active_day_share = float((ready["usage"] > 0).mean())
        records.append(
            {
                "city": city,
                "all_local_days": int(len(group)),
                "analysis_days": int(len(ready)),
                "excluded_days": int((~group["analysis_ready"]).sum()),
                "zero_usage_days": int((ready["usage"] == 0).sum()),
                "active_day_share": active_day_share,
                "continuous_system": bool(active_day_share >= 0.99),
                "usage_total": float(ready["usage"].sum()),
                "usage_daily_median": float(ready["usage"].median()),
                "utci_missing_hours": int((group["observed_hours"] - group["utci_observed_hours"]).sum()),
                "utci_mean_c": float(ready["utci_mean_c"].mean()),
                "utci_min_c": float(ready["utci_min_c"].min()),
                "utci_max_c": float(ready["utci_max_c"].max()),
                "precipitation_total_mm": float(ready["precipitation_mm"].sum()),
            }
        )
    return records


def main():
    metadata = pd.read_csv(METADATA_PATH)
    stock = pd.read_csv(
        STOCK_PATH,
        header=None,
        names=["city_key", "source_date", "source_hour", "usage"],
    )
    utci = pd.read_csv(UTCI_PATH, index_col=0)
    rain = pd.read_csv(RAIN_PATH, index_col=0)

    require(metadata.shape == (40, 6), "metadata shape must be 40 by 6")
    require(utci.shape == (40, 8760), "UTCI shape must be 40 by 8760")
    require(rain.shape == (40, 8760), "rain shape must be 40 by 8760")
    require(metadata["City"].tolist() == utci.index.tolist(), "UTCI row order must match metadata")
    require(metadata["City"].tolist() == rain.index.tolist(), "rain row order must match metadata")
    require(stock["usage"].notna().all(), "stock usage contains missing values")
    require((stock["usage"] >= 0).all(), "stock usage contains negative values")

    selected_metadata = metadata.iloc[SELECTED_ROWS].copy()
    selected_keys = set(selected_metadata["City"].str.lower())
    stock_keys = set(stock["city_key"].unique())
    require(selected_keys.issubset(stock_keys), "selected cities are missing from stock data")
    extra_stock_cities = stock_keys - selected_keys
    require(extra_stock_cities == EXPECTED_EXTRA_STOCK_CITIES, "unexpected stock city set")

    selected_stock = stock.loc[stock["city_key"].isin(selected_keys)].copy()
    row_counts = selected_stock.groupby("city_key").size()
    require((row_counts == 8760).all(), "every selected city must have 8760 stock rows")

    hourly_frames = [
        build_city_hourly(city_row, selected_stock, utci, rain)
        for _, city_row in selected_metadata.iterrows()
    ]
    hourly = pd.concat(hourly_frames, ignore_index=True)
    daily = aggregate_daily(hourly)
    output = daily.loc[daily["analysis_ready"]].copy()
    output["usage_log1p"] = np.log1p(output["usage"])

    city_active_share = output.groupby("city")["usage"].transform(lambda values: (values > 0).mean())
    output["active_day_share"] = city_active_share
    output["continuous_system"] = (city_active_share >= 0.99).astype(int)

    city_means = output.groupby("city")["utci_mean_c"].transform("mean")
    city_sds = output.groupby("city")["utci_mean_c"].transform("std")
    output["utci_city_mean_c"] = city_means
    output["utci_anomaly_10c"] = (output["utci_mean_c"] - city_means) / 10.0
    output["utci_anomaly_sq"] = output["utci_anomaly_10c"] ** 2
    output["utci_within_city_z"] = (output["utci_mean_c"] - city_means) / city_sds
    output["precipitation_log1p"] = np.log1p(output["precipitation_mm"])

    output = output[
        [
            "city",
            "country",
            "climate_code",
            "latitude",
            "longitude",
            "timezone",
            "date",
            "usage",
            "usage_log1p",
            "active_day_share",
            "continuous_system",
            "utci_mean_c",
            "utci_min_c",
            "utci_max_c",
            "utci_city_mean_c",
            "utci_anomaly_10c",
            "utci_anomaly_sq",
            "utci_within_city_z",
            "precipitation_mm",
            "precipitation_log1p",
            "weekday",
            "weekend",
            "day_of_year",
            "season_sin",
            "season_cos",
            "observed_hours",
            "utci_observed_hours",
            "rain_observed_hours",
        ]
    ].sort_values(["city", "date"], ignore_index=True)

    require(not output.duplicated(["city", "date"]).any(), "daily keys must be unique")
    require(output.isna().sum().sum() == 0, "analysis output contains missing values")
    require((output["usage"] >= 0).all(), "daily usage contains negative values")
    require((output["precipitation_mm"] >= 0).all(), "daily rain contains negative values")
    require(output["city"].nunique() == 29, "analysis output must contain 29 cities")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")

    audit = {
        "source_files": {
            str(path.relative_to(ROOT)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in [ARCHIVE_PATH, METADATA_PATH, STOCK_PATH, UTCI_PATH, RAIN_PATH]
        },
        "source_shapes": {
            "metadata": [int(value) for value in metadata.shape],
            "stock": [int(value) for value in stock.shape],
            "utci": [int(value) for value in utci.shape],
            "rain": [int(value) for value in rain.shape],
        },
        "selection": {
            "metadata_row_numbers_one_based": [value + 1 for value in SELECTED_ROWS],
            "selected_city_count": int(len(selected_metadata)),
            "selected_cities": selected_metadata["City"].tolist(),
            "excluded_stock_cities": sorted(extra_stock_cities),
        },
        "stock_quality": {
            "selected_hourly_rows": int(len(selected_stock)),
            "missing_usage": int(selected_stock["usage"].isna().sum()),
            "negative_usage": int((selected_stock["usage"] < 0).sum()),
            "fractional_usage_rows": int((selected_stock["usage"] % 1 != 0).sum()),
            "zero_usage_rows": int((selected_stock["usage"] == 0).sum()),
            "duplicate_source_keys": duplicate_key_records(selected_stock),
        },
        "daily_quality": {
            "all_local_days": int(len(daily)),
            "analysis_rows": int(len(output)),
            "excluded_days": int((~daily["analysis_ready"]).sum()),
            "duplicate_city_dates": int(output.duplicated(["city", "date"]).sum()),
            "missing_cells": int(output.isna().sum().sum()),
            "zero_usage_days": int((output["usage"] == 0).sum()),
            "continuous_city_count": int(
                output.loc[output["continuous_system"] == 1, "city"].nunique()
            ),
            "primary_cities": sorted(
                output.loc[output["continuous_system"] == 1, "city"].unique().tolist()
            ),
            "primary_sample_rows": int((output["continuous_system"] == 1).sum()),
            "date_min": output["date"].min().date().isoformat(),
            "date_max": output["date"].max().date().isoformat(),
            "observed_hours_counts": {
                str(int(key)): int(value)
                for key, value in output["observed_hours"].value_counts().sort_index().items()
            },
            "excluded_reason_counts_nonexclusive": {
                "outside_nominal_period": int((~daily["inside_nominal_period"]).sum()),
                "hours_outside_23_to_25": int((~daily["observed_hours"].between(23, 25)).sum()),
                "fewer_than_20_utci_hours": int((daily["utci_observed_hours"] < 20).sum()),
                "incomplete_rain": int(
                    (daily["rain_observed_hours"] != daily["observed_hours"]).sum()
                ),
            },
        },
        "processing_rules": {
            "hour_alignment": "Preserve each city stock row order and map 8760 rows to consecutive UTC hours from its first source date.",
            "local_day": "Convert UTC hours to the metadata time zone before daily aggregation.",
            "daily_usage": "Sum all estimated hourly hires, including zero values.",
            "daily_utci": "Use the mean, minimum, and maximum of available hourly UTCI values.",
            "daily_precipitation": "Sum hourly precipitation after conversion from metres to millimetres.",
            "analysis_day": "Keep nominal-period local days with 23 to 25 hourly rows, at least 20 UTCI values, and complete precipitation.",
            "primary_sample": "Use cities with positive estimated usage on at least 99 percent of analysis days.",
        },
        "per_city": city_summary(daily),
        "output": {
            "path": str(OUTPUT_PATH.relative_to(ROOT)),
            "sha256": sha256(OUTPUT_PATH),
            "rows": int(len(output)),
            "columns": int(len(output.columns)),
        },
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(audit["daily_quality"], indent=2))
    print(json.dumps(audit["stock_quality"]["duplicate_source_keys"], indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {AUDIT_PATH}")


if __name__ == "__main__":
    main()
