import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import WORK_ROOT, read_json
import figures as fig


def available(run_id, model, fold):
    return (WORK_ROOT / "runs" / run_id / model / fold / "predictions.csv").exists()


def choose_trajectory_cities(run_id, model, fold):
    directory = WORK_ROOT / "runs" / run_id / model / fold
    per_city = pd.read_csv(directory / "metrics_city.csv")
    per_city = per_city.loc[per_city["task"] == "block"].sort_values("wis_log")
    ordered = per_city["city"].tolist()
    median_city = ordered[len(ordered) // 2]
    missing = [city for city in ("Paris", "Lund", median_city) if city not in ordered]
    if missing:
        raise ValueError(f"cities missing from the prediction metrics: {missing}")
    return ["Paris", "Lund", median_city], median_city


def choose_temperature_cities(run_id, model, fold):
    directory = WORK_ROOT / "runs" / run_id / model / fold
    payload = read_json(directory / "effects.json")
    cities = pd.DataFrame(payload["cities"])
    warm = cities.sort_values("observed_utci_max_c", ascending=False)["city"].iloc[0]
    cold = cities.sort_values("observed_utci_min_c")["city"].iloc[0]
    picks = ["Paris"]
    for candidate in (warm, cold):
        if candidate not in picks:
            picks.append(candidate)
    return picks[:3]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="formal_20260930")
    parser.add_argument("--fold", default="dev1")
    parser.add_argument("--final-fold", default="final")
    args = parser.parse_args(argv)

    selection = read_json(WORK_ROOT / "tables" / "selection.json")
    selected = selection["selected_model"]
    produced = []
    produced.append(fig.figure_city_overview())

    effect_paths = [WORK_ROOT / "runs" / args.run_id / "M0" / "full" / "effects.json",
                    WORK_ROOT / "runs" / args.run_id / selected / "full" / "effects.json"]
    if all(path.exists() for path in effect_paths):
        produced.append(fig.figure_rain_forest(effect_paths))
    else:
        raise FileNotFoundError("full-period effects are required for the forest plot")

    temperature_cities = choose_temperature_cities(args.run_id, selected, "full")
    produced.append(fig.figure_temperature_curves(
        args.run_id, selected, "full", temperature_cities
    ))

    entries = [("M0 residual", "M0", args.fold, fig.GREY),
               ("M1 residual", "M1", args.fold, fig.GOLD),
               ("M2 innovation", "M2", args.fold, fig.PURPLE)]
    entries = [item for item in entries if (WORK_ROOT / "runs" / args.run_id / item[1] / item[2]
                                            / "residual_acf.csv").exists()]
    if entries:
        produced.append(fig.figure_residual_acf(args.run_id, entries))

    models = [name for name in ("M0", "M1", "M2") if available(args.run_id, name, args.fold)]
    if selected not in models:
        models.append(selected)
    produced.append(fig.figure_forecast_scores(args.run_id, sorted(set(models))))

    trajectory_cities, median_city = choose_trajectory_cities(args.run_id, selected, args.final_fold)
    produced.append(fig.figure_city_trajectories(
        args.run_id, selected, args.final_fold, trajectory_cities
    ))

    if (WORK_ROOT / "runs" / args.run_id / "A0" / args.fold / "posterior_summary.csv").exists():
        produced.append(fig.figure_pooling_ablation(args.run_id, args.fold))

    prior_path = WORK_ROOT / "runs" / args.run_id / selected / args.fold / "prior_predictive.json"
    if prior_path.exists():
        produced.append(fig.figure_prior_predictive(args.run_id, selected, args.fold))

    produced.append(fig.figure_all_city_curves(args.run_id, selected, "full"))

    convergence_entries = [(name, name, args.fold) for name in ("M0", "M1", "M2")]
    convergence_entries = [item for item in convergence_entries
                           if (WORK_ROOT / "runs" / args.run_id / item[1] / item[2]
                               / "posterior_summary.csv").exists()]
    if convergence_entries:
        produced.append(fig.figure_convergence(args.run_id, convergence_entries))

    sensitivity_root = WORK_ROOT / "runs" / args.run_id
    if (sensitivity_root / f"{selected}_s05" / args.fold / "effects.json").exists():
        produced.append(fig.figure_sensitivity(args.run_id, selected, args.fold))
    ppc_path = WORK_ROOT / "runs" / args.run_id / selected / args.fold / "ppc_train.csv"
    if ppc_path.exists():
        produced.append(fig.figure_posterior_predictive(args.run_id, selected, args.fold))

    produced.append(fig.figure_prediction_attribution(args.run_id))

    for item in produced:
        print(item)
    print(f"trajectory cities: {trajectory_cities} (median performer {median_city})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
