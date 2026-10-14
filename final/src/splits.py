import json
from pathlib import Path

import pandas as pd

from data import CITY, DATE, apply_preprocessing, build_split, fit_preprocessing, roster_prediction

SPLITS_FILE = Path(__file__).resolve().parents[1] / "configs" / "splits.json"


def load_splits(path=None):
    target = Path(path) if path else SPLITS_FILE
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("roster") != "prediction":
        raise ValueError("splits configuration must target the prediction roster")
    for name, fold in payload["folds"].items():
        for key in ("cutoff", "test_start", "test_end", "role"):
            if key not in fold:
                raise ValueError(f"fold {name} missing key {key}")
        if not (pd.Timestamp(fold["cutoff"]) < pd.Timestamp(fold["test_start"]) <= pd.Timestamp(fold["test_end"])):
            raise ValueError(f"fold {name} has inconsistent dates")
    return payload


def fold_windows(payload):
    windows = {}
    for name, fold in payload["folds"].items():
        windows[name] = {
            "cutoff": pd.Timestamp(fold["cutoff"]),
            "test_start": pd.Timestamp(fold["test_start"]),
            "test_end": pd.Timestamp(fold["test_end"]),
            "role": fold["role"],
        }
    return windows


def make_split_index(frame, payload, city_names=None):
    names = city_names if city_names is not None else roster_prediction(frame)
    records = []
    for name, window in fold_windows(payload).items():
        subset = frame.loc[frame[CITY].isin(names)]
        part = pd.Series("train", index=subset.index)
        part.loc[subset[DATE] > window["cutoff"]] = "after_cutoff"
        part.loc[
            (subset[DATE] >= window["test_start"]) & (subset[DATE] <= window["test_end"])
        ] = "test"
        block = subset[[CITY, DATE]].copy()
        block["fold"] = name
        block["split"] = part.to_numpy()
        records.append(block)
    index = pd.concat(records, ignore_index=True)
    return index


def prepare_fold(frame, payload, fold_name, city_names=None):
    names = city_names if city_names is not None else roster_prediction(frame)
    window = fold_windows(payload)[fold_name]
    bundle = build_split(frame, names, window["cutoff"], window["test_start"], window["test_end"])
    bundle["fold"] = fold_name
    bundle["window"] = window
    return bundle


def prepare_full_period(frame, city_names):
    names = list(city_names)
    subset = frame.loc[frame[CITY].isin(names)]
    if subset.empty:
        raise ValueError("no rows for the requested roster")
    prep = fit_preprocessing(subset, names)
    train_frame, train_arrays = apply_preprocessing(subset, prep)
    return {
        "prep": prep,
        "train_frame": train_frame,
        "train_arrays": train_arrays,
        "test_frame": None,
        "test_arrays": None,
        "fold": "full",
        "window": {
            "cutoff": str(subset[DATE].max().date()),
            "test_start": None,
            "test_end": None,
            "role": "full_period_association",
        },
    }


def assert_no_leakage(bundle):
    train = bundle["train_frame"]
    test = bundle["test_frame"]
    cutoff = bundle["window"]["cutoff"]
    if train[DATE].max() > cutoff:
        raise ValueError("training partition contains dates after the cutoff")
    if test[DATE].min() <= cutoff:
        raise ValueError("test partition contains dates at or before the cutoff")
    if set(train[CITY]) != set(test[CITY]):
        raise ValueError("train and test city rosters differ")
    return True
