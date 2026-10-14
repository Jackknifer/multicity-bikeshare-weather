import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import WORK_ROOT, read_json, write_json

TASKS = ("block", "one_step")
DEVELOPMENT_FOLDS = ("dev1", "dev2")
WIS_TOLERANCE = 0.02


def collect_model_metrics(run_id):
    records = []
    root = WORK_ROOT / "runs" / run_id
    if not root.is_dir():
        raise FileNotFoundError(f"run directory not found: {root}")
    for path in sorted(root.glob("*/*/metrics_overall.json")):
        payload = read_json(path)
        if "tasks" not in payload:
            continue
        run_name = payload.get("run_name", payload["model"])
        if run_name != payload["model"]:
            continue
        diagnostics = read_json(path.with_name("diagnostics.json"))
        for task, summary in payload["tasks"].items():
            records.append({
                "model": payload["model"],
                "fold": payload["fold"],
                "task": task,
                "diagnostics_passed": diagnostics["passed"],
                "parameters": diagnostics["parameters"],
                "max_rhat": diagnostics["max_rhat"],
                "min_ess_bulk": diagnostics["min_ess_bulk"],
                "min_ess_tail": diagnostics["min_ess_tail"],
                "divergences": diagnostics["divergences"],
                "maxdepth_reached_sum": diagnostics.get("maxdepth_reached_sum", 0),
                "negative_median_log_fraction": payload["negative_median_log_fraction"],
                **{key: value for key, value in summary.items() if key != "worst_city_wis"},
            })
    if not records:
        raise FileNotFoundError(f"no fitted-model metrics under {root}")
    return pd.DataFrame(records)


def collect_baseline_metrics(run_id):
    records = []
    root = WORK_ROOT / "runs" / run_id / "baselines"
    if not root.is_dir():
        raise FileNotFoundError(f"baseline directory not found: {root}")
    for path in sorted(root.glob("*/metrics_overall.json")):
        payload = read_json(path)
        for key, summary in payload["models"].items():
            model, task = key.split("|")
            records.append({
                "model": model,
                "fold": payload["fold"],
                "task": task,
                "diagnostics_passed": True,
                "parameters": 0,
                "max_rhat": np.nan,
                "min_ess_bulk": np.nan,
                "min_ess_tail": np.nan,
                "divergences": 0,
                "maxdepth_reached_sum": 0,
                "negative_median_log_fraction": np.nan,
                **{k: v for k, v in summary.items() if k != "worst_city_wis"},
            })
    if not records:
        raise FileNotFoundError(f"no baseline metrics under {root}")
    return pd.DataFrame(records)


def selection_table(metrics, candidates=("M0", "M1", "M2")):
    block = metrics.loc[(metrics["task"] == "block") & metrics["model"].isin(candidates)]
    dev = block.loc[block["fold"].isin(DEVELOPMENT_FOLDS)]
    if dev.empty:
        raise ValueError("no development-fold block metrics available")
    pivot = dev.pivot_table(
        index="model",
        columns="fold",
        values=["city_equal_wis_log", "city_equal_mean_log_predictive_density"],
        aggfunc="mean",
    )
    table = pd.DataFrame(index=sorted(candidates))
    table["dev_mean_wis"] = pivot["city_equal_wis_log"].mean(axis=1).reindex(table.index)
    table["dev_mean_density"] = pivot[
        "city_equal_mean_log_predictive_density"
    ].mean(axis=1).reindex(table.index)
    for fold in DEVELOPMENT_FOLDS:
        table[f"wis_{fold}"] = pivot["city_equal_wis_log"][fold].reindex(table.index)
        table[f"density_{fold}"] = pivot[
            "city_equal_mean_log_predictive_density"
        ][fold].reindex(table.index)
    table["all_diagnostics_passed"] = [
        bool(metrics.loc[(metrics["model"] == name) & metrics["fold"].isin(DEVELOPMENT_FOLDS),
                         "diagnostics_passed"].all())
        for name in table.index
    ]
    best = float(table["dev_mean_wis"].min())
    table["relative_gap_to_best"] = table["dev_mean_wis"] / best - 1.0
    table["within_tolerance"] = table["relative_gap_to_best"] <= WIS_TOLERANCE
    eligible = table.loc[table["within_tolerance"] & table["all_diagnostics_passed"]]
    if eligible.empty:
        raise ValueError("no candidate passes diagnostics within the WIS tolerance")
    ranked = eligible.sort_values(
        ["dev_mean_density", "dev_mean_wis"], ascending=[False, True]
    )
    selected = ranked.index[0]
    return table, selected


def paired_bootstrap(frame_a, frame_b, block_days, draws, seed):
    merged = frame_a[["city", "date", "wis_log"]].merge(
        frame_b[["city", "date", "wis_log"]], on=["city", "date"], suffixes=("_a", "_b")
    )
    if merged.empty:
        raise ValueError("no overlapping rows for the paired comparison")
    merged = merged.sort_values(["date", "city"]).reset_index(drop=True)
    difference = merged["wis_log_a"].to_numpy() - merged["wis_log_b"].to_numpy()
    city_codes = pd.Categorical(merged["city"]).codes
    dates = np.sort(merged["date"].unique())
    blocks = [np.flatnonzero(merged["date"].to_numpy() == date) for date in dates]
    grouped = [np.concatenate(blocks[i:i + block_days]) for i in range(0, len(blocks), block_days)]
    city_count = int(city_codes.max()) + 1
    rng = np.random.default_rng(seed)
    statistics = np.empty(draws)
    for draw in range(draws):
        chosen = rng.integers(0, len(grouped), len(grouped))
        index = np.concatenate([grouped[item] for item in chosen])
        sums = np.bincount(city_codes[index], weights=difference[index], minlength=city_count)
        counts = np.bincount(city_codes[index], minlength=city_count)
        statistics[draw] = float(np.mean(sums / counts))
    observed = float(merged.groupby("city")["wis_log_a"].mean().mean()
                     - merged.groupby("city")["wis_log_b"].mean().mean())
    return {
        "observed_city_equal_wis_difference": observed,
        "bootstrap_mean": float(statistics.mean()),
        "q025": float(np.quantile(statistics, 0.025)),
        "q975": float(np.quantile(statistics, 0.975)),
        "block_days": block_days,
        "draws": draws,
        "note": "positive values mean the first model has the larger weighted interval score",
    }


def paired_comparison(run_id, model_a, model_b, task="block", block_days=(7, 14), draws=2000):
    results = {}
    for fold in DEVELOPMENT_FOLDS:
        frame_a = pd.read_csv(WORK_ROOT / "runs" / run_id / model_a / fold / "predictions.csv",
                              parse_dates=["date"])
        frame_b = pd.read_csv(WORK_ROOT / "runs" / run_id / model_b / fold / "predictions.csv",
                              parse_dates=["date"])
        frame_a = frame_a.loc[frame_a["task"] == task]
        frame_b = frame_b.loc[frame_b["task"] == task]
        results[fold] = {
            f"block_{days}": paired_bootstrap(frame_a, frame_b, days, draws, 27)
            for days in block_days
        }
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="formal_20260930")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else WORK_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_metrics = collect_model_metrics(args.run_id)
    baseline_metrics = collect_baseline_metrics(args.run_id)
    combined = pd.concat([model_metrics, baseline_metrics], ignore_index=True)
    combined.to_csv(out_dir / "all_metrics.csv", index=False)

    table, selected = selection_table(model_metrics)
    table.to_csv(out_dir / "selection_dev_folds.csv")
    write_json(out_dir / "selection.json", {
        "run_id": args.run_id,
        "criterion": "city-equal weighted interval score on the log1p scale, block task",
        "development_folds": list(DEVELOPMENT_FOLDS),
        "wis_relative_tolerance": WIS_TOLERANCE,
        "selected_model": selected,
        "table": table.reset_index().rename(columns={"index": "model"}).to_dict(orient="records"),
    })

    final_models = sorted(set(model_metrics.loc[model_metrics["fold"] == "final", "model"]))
    if final_models:
        final_table = model_metrics.loc[model_metrics["fold"] == "final"].pivot_table(
            index="model", columns="task",
            values=["city_equal_wis_log", "city_equal_mean_log_predictive_density",
                    "city_equal_coverage_90", "normalised_mae_log"],
        )
        final_table.to_csv(out_dir / "final_fold_metrics.csv")

    baseline_table = baseline_metrics.pivot_table(
        index=["fold", "task", "model"],
        values=["city_equal_wis_log", "city_equal_mean_log_predictive_density",
                "city_equal_coverage_90", "normalised_mae_log", "mae_usage"],
    )
    baseline_table.to_csv(out_dir / "baseline_metrics.csv")

    if selected != "M0" and all(
        (WORK_ROOT / "runs" / args.run_id / name / fold / "predictions.csv").exists()
        for name in (selected, "M0") for fold in DEVELOPMENT_FOLDS
    ):
        comparison = paired_comparison(args.run_id, selected, "M0", task="block")
        write_json(out_dir / "paired_comparison.json", {
            "run_id": args.run_id,
            "task": "block",
            "first_model": selected,
            "second_model": "M0",
            "folds": comparison,
        })

    print(f"selected model: {selected}")
    print(table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
