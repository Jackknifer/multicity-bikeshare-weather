import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import WORK_ROOT, read_json

GENERATED = WORK_ROOT / "report" / "generated"
BASELINE_NAMES = {
    "city_constant": "City constant",
    "seasonal_naive_7": "Seasonal naive, 7-day lag",
    "city_ols": "City OLS, parameter uncertainty",
    "city_ar_ols": "City AR-OLS",
}
BASELINE_ORDER = ("city_constant", "seasonal_naive_7", "city_ols", "city_ar_ols")


def fmt(value, digits=2):
    if value is None:
        return "--"
    value = float(value)
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def percent(value, digits=2):
    return f"{100.0 * value:.{digits}f}"


DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
               "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def latex_name(name):
    return "".join(DIGIT_WORDS.get(character, character) for character in name)


def macro(name, value):
    return f"\\newcommand{{\\{latex_name(name)}}}{{{value}}}"


def interval_macros(prefix, payload):
    return [
        macro(f"{prefix}Med", fmt(payload["median"])),
        macro(f"{prefix}Lo", fmt(payload["q025"])),
        macro(f"{prefix}Hi", fmt(payload["q975"])),
    ]


def tabular(column_format, header, rows):
    lines = [
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    lines.extend(rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_tables(metrics, run_id):
    tables = {}

    baselines = metrics.loc[metrics["model"].isin(BASELINE_ORDER)]
    rows = []
    compact_rows = []
    for fold in ("dev1", "dev2", "final"):
        for task in ("block", "one_step"):
            for model in BASELINE_ORDER:
                block = baselines.loc[(baselines["model"] == model) & (baselines["fold"] == fold)
                                      & (baselines["task"] == task)]
                if block.empty:
                    continue
                row = block.iloc[0]
                line = (
                    f"{fold} & {task.replace('_', ' ')} & {BASELINE_NAMES[model]} & "
                    f"{fmt(row['city_equal_wis_log'], 4)} & "
                    f"{fmt(row['city_equal_mean_log_predictive_density'], 3)} & "
                    f"{percent(row['city_equal_coverage_90'], 1)} & "
                    f"{fmt(row['normalised_mae_log'], 3)} \\\\"
                )
                rows.append(line)
                if fold in ("dev1", "final"):
                    compact_rows.append(line)
    header = "Fold & Task & Baseline & WIS & Log density & Coverage 90 (\\%) & Norm.\\ MAE"
    tables["table_baselines.tex"] = tabular("lllrrrr", header, rows)
    tables["table_baselines_compact.tex"] = tabular("lllrrrr", header, compact_rows)

    models = metrics.loc[metrics["model"].str.startswith(("M", "A"))]
    rows = []
    for fold in ("dev1", "dev2", "final"):
        for model in ("M0", "M1", "M2", "A0", "A1"):
            block = models.loc[(models["model"] == model) & (models["fold"] == fold)
                               & (models["task"] == "block")]
            if block.empty:
                continue
            row = block.iloc[0]
            rows.append(
                f"{fold} & {model} & {fmt(row['city_equal_wis_log'], 4)} & "
                f"{fmt(row['city_equal_mean_log_predictive_density'], 3)} & "
                f"{percent(row['city_equal_coverage_90'], 1)} & "
                f"{fmt(row['normalised_mae_log'], 3)} & "
                f"{fmt(row['min_ess_bulk'], 0)} & {fmt(row['max_rhat'], 4)} \\\\"
            )
    tables["table_models.tex"] = tabular(
        "llrrrrrr",
        "Fold & Model & WIS & Log density & Coverage 90 (\\%) & Norm.\\ MAE & Min ESS & Max $\\hat{R}$",
        rows,
    )

    full_rows = []
    for model in ("M0", "M1", "M2", "A0", "A1"):
        path = WORK_ROOT / "runs" / run_id / model / "full" / "effects.json"
        if not path.exists():
            continue
        payload = read_json(path)
        rain = payload["rain_10mm_city_equal_percent"]
        temp = payload["temp_10c_city_equal_percent"]
        full_rows.append(
            f"{model} & {fmt(rain['median'])} [{fmt(rain['q025'])}, {fmt(rain['q975'])}] & "
            f"{fmt(temp['median'])} [{fmt(temp['q025'])}, {fmt(temp['q975'])}] & "
            f"{payload['cities_with_interior_maximum_probability_above_0_95']} \\\\"
        )
    if full_rows:
        tables["table_effects.tex"] = tabular(
            "lrrr",
            "Model & Precipitation $+10$ mm (\\%) & UTCI $+10^\\circ$C (\\%) & Interior peak $>0.95$",
            full_rows,
        )
    return tables


def shrink_summary(run_id, fold):
    frames = {}
    for model in ("A0", "A1"):
        path = WORK_ROOT / "runs" / run_id / model / fold / "posterior_summary.csv"
        if not path.exists():
            return None
        summary = pd.read_csv(path)
        block = summary.loc[summary["parameter"].str.startswith("beta_rain_city[")].copy()
        block["city"] = block["parameter"].str.extract(r"\[(.*)\]")
        frames[model] = block.set_index("city")["sd"]
    joined = pd.DataFrame(frames).dropna()
    reduction = (joined["A0"] - joined["A1"]) / joined["A0"]
    return {
        "cities": int((joined["A1"] < joined["A0"]).sum()),
        "total": int(len(joined)),
        "median_reduction_percent": float(np.median(reduction) * 100.0),
    }


def build_macros(metrics, selection, run_id):
    selected = selection["selected_model"]
    lines = [macro("SelModel", selected)]

    for model in ("M0", "M1", "M2"):
        for fold in ("dev1", "dev2", "final"):
            block = metrics.loc[(metrics["model"] == model) & (metrics["fold"] == fold)
                                & (metrics["task"] == "block")]
            if block.empty:
                continue
            row = block.iloc[0]
            tag = f"{model}{fold.capitalize()}"
            lines.append(macro(f"{tag}Wis", fmt(row["city_equal_wis_log"], 4)))
            lines.append(macro(f"{tag}Density", fmt(row["city_equal_mean_log_predictive_density"], 3)))
            lines.append(macro(f"{tag}Cov", percent(row["city_equal_coverage_90"], 1)))
            lines.append(macro(f"{tag}NormMae", fmt(row["normalised_mae_log"], 3)))
            lines.append(macro(f"{tag}MinEss", fmt(row["min_ess_bulk"], 0)))
            lines.append(macro(f"{tag}MaxRhat", fmt(row["max_rhat"], 5)))

    for fold in ("dev1", "dev2", "final"):
        for task in ("block", "one_step"):
            for model in dict.fromkeys(("M0", "M1", selected)):
                block = metrics.loc[(metrics["model"] == model) & (metrics["fold"] == fold)
                                    & (metrics["task"] == task)]
                if block.empty:
                    continue
                row = block.iloc[0]
                tag = f"{model}{fold.capitalize()}{'Block' if task == 'block' else 'Step'}"
                lines.append(macro(f"{tag}Wis", fmt(row["city_equal_wis_log"], 4)))
                lines.append(macro(f"{tag}Density", fmt(row["city_equal_mean_log_predictive_density"], 3)))
                lines.append(macro(f"{tag}Cov", percent(row["city_equal_coverage_90"], 1)))
                lines.append(macro(f"{tag}Width", fmt(row["width_90"], 3)))

    for fold, suffix in (("dev1", "DevOne"), ("dev2", "DevTwo"), ("final", "Final")):
        for task, task_tag in (("block", "Block"), ("one_step", "Step")):
            block = metrics.loc[(metrics["model"] == "M1nt") & (metrics["fold"] == fold)
                                & (metrics["task"] == task)]
            if block.empty:
                continue
            row = block.iloc[0]
            lines.append(macro(f"NoTrend{suffix}{task_tag}Wis",
                               fmt(row["city_equal_wis_log"], 4)))
            lines.append(macro(f"NoTrend{suffix}{task_tag}Density",
                               fmt(row["city_equal_mean_log_predictive_density"], 3)))
            lines.append(macro(f"NoTrend{suffix}{task_tag}Cov",
                               percent(row["city_equal_coverage_90"], 1)))

    for fold in ("dev1", "dev2"):
        for model in ("M0", "M1", "M2", "A0", "A1"):
            block = metrics.loc[(metrics["model"] == model) & (metrics["fold"] == fold)
                                & (metrics["task"] == "block")]
            if block.empty:
                continue
            lines.append(macro(f"{model}{fold.capitalize()}BlockWis", fmt(block.iloc[0]["city_equal_wis_log"], 4)))

    naive = metrics.loc[(metrics["model"] == "seasonal_naive_7") & (metrics["fold"] == "dev1")]
    for task, suffix in (("block", "Block"), ("one_step", "Step")):
        block = naive.loc[naive["task"] == task]
        if not block.empty:
            lines.append(macro(f"BaselineNaive{suffix}Wis", fmt(block.iloc[0]["city_equal_wis_log"], 4)))
    ols = metrics.loc[(metrics["model"] == "city_ols") & (metrics["fold"] == "dev1")
                      & (metrics["task"] == "block")]
    if not ols.empty:
        lines.append(macro("BaselineOlsCov", percent(ols.iloc[0]["city_equal_coverage_90"], 2)))

    for name, model in (("M0", "M0"), ("Sel", selected)):
        path = WORK_ROOT / "runs" / run_id / model / "full" / "effects.json"
        if not path.exists():
            continue
        payload = read_json(path)
        lines += interval_macros(f"Rain{name}", payload["rain_10mm_city_equal_percent"])
        lines += interval_macros(f"Temp{name}", payload["temp_10c_city_equal_percent"])
        lines += interval_macros(f"RainSupport{name}", payload["rain_support_city_equal_percent"])
        lines += interval_macros(f"TempSupport{name}", payload["temp_support_city_equal_percent"])
        lines.append(macro(f"Rain{name}PNeg", fmt(payload["p_population_rain_negative"], 3)))
        lines.append(macro(f"PeakCities{name}",
                           payload["cities_with_interior_maximum_probability_above_0_95"]))
        lines.append(macro(f"ExtrapCities{name}",
                           payload["cities_with_temp_10c_outside_support"]))
        if name == "Sel":
            rain_cities = pd.DataFrame(payload["cities"])
            lines.append(macro("RainCityMin", fmt(rain_cities["rain_10mm_percent"]
                                                  .apply(lambda item: item["median"]).min())))
            lines.append(macro("RainCityMax", fmt(rain_cities["rain_10mm_percent"]
                                                  .apply(lambda item: item["median"]).max())))
            lines.append(macro("CityCount", int(len(rain_cities))))

    for model in ("M0", "M1", "M2"):
        for fold in ("dev1", "dev2", "final"):
            path = WORK_ROOT / "runs" / run_id / model / fold / "residual_summary.json"
            if not path.exists():
                continue
            payload = read_json(path)
            tag = f"{model}{fold.capitalize()}"
            lines.append(macro(f"{tag}LagOne", fmt(payload["lag1_median"], 3)))
            lines.append(macro(f"{tag}LagOneMax", fmt(payload["lag1_max"], 3)))
            lines.append(macro(f"{tag}LagOneAbove", payload["cities_lag1_above_0_5"]))
            lines.append(macro(f"{tag}TrainCov", percent(payload["training_coverage_90"], 2)))

    path = WORK_ROOT / "runs" / run_id / selected / "dev1" / "prior_predictive.json"
    if path.exists():
        payload = read_json(path)
        lines.append(macro("PriorNegative",
                           percent(payload["intercept_negative_fraction"], 2)))
        quantiles = payload["rain_10mm_percent_quantiles"]
        lines.append(macro("PriorRainLow", fmt(quantiles["0.05"], 1)))
        lines.append(macro("PriorRainHigh", fmt(quantiles["0.95"], 1)))
        lines.append(macro("PriorTempLow", fmt(payload["temp_10c_percent_quantiles"]["0.05"], 1)))
        lines.append(macro("PriorTempHigh", fmt(payload["temp_10c_percent_quantiles"]["0.95"], 1)))
        lines.append(macro("PriorPhiLow", fmt(payload["phi_quantiles"]["0.05"], 2)))
        lines.append(macro("PriorPhiHigh", fmt(payload["phi_quantiles"]["0.95"], 2)))
    path = WORK_ROOT / "runs" / run_id / selected / "dev1" / "residual_summary.json"
    if path.exists():
        payload = read_json(path)
        lines.append(macro("TrainNegative",
                           percent(payload["negative_log_prediction_fraction"], 3)))

    shrink = shrink_summary(run_id, "dev1")
    if shrink:
        lines.append(macro("ShrinkCities", f"{shrink['cities']} of {shrink['total']}"))
        lines.append(macro("ShrinkMedian", fmt(shrink["median_reduction_percent"], 1)))

    ablation_a0 = metrics.loc[(metrics["model"] == "A0") & (metrics["fold"] == "dev1")
                              & (metrics["task"] == "block")]
    ablation_a1 = metrics.loc[(metrics["model"] == "A1") & (metrics["fold"] == "dev1")
                              & (metrics["task"] == "block")]
    if not ablation_a0.empty and not ablation_a1.empty:
        value_a0 = float(ablation_a0.iloc[0]["city_equal_wis_log"])
        value_a1 = float(ablation_a1.iloc[0]["city_equal_wis_log"])
        lines.append(macro("AblationWisAzero", fmt(value_a0, 4)))
        lines.append(macro("AblationWisAone", fmt(value_a1, 4)))
        lines.append(macro("AblationWisGap", fmt(100.0 * (value_a1 / value_a0 - 1.0), 2)))

    if selected != "M0":
        path = WORK_ROOT / "tables" / "paired_comparison.json"
        if path.exists():
            payload = read_json(path)
            for fold in ("dev1", "dev2"):
                if fold not in payload["folds"]:
                    continue
                for block in (7, 14):
                    entry = payload["folds"][fold][f"block_{block}"]
                    tag = f"Pair{fold.capitalize()}B{block}"
                    lines.append(macro(f"{tag}Obs",
                                       fmt(entry["observed_city_equal_wis_difference"], 4)))
                    lines.append(macro(f"{tag}Lo", fmt(entry["q025"], 4)))
                    lines.append(macro(f"{tag}Hi", fmt(entry["q975"], 4)))

    for item in selection["table"]:
        lines.append(macro(f"Sel{item['model']}DevWis", fmt(item["dev_mean_wis"], 4)))
        lines.append(macro(f"Sel{item['model']}DevDensity", fmt(item["dev_mean_density"], 3)))
        lines.append(macro(f"Sel{item['model']}Gap", percent(item["relative_gap_to_best"], 2)))

    core = metrics.loc[metrics["model"].str.match(r"^M[0-9]$|^A[0-9]$")]
    if not core.empty:
        lines.append(macro("CoreMinEss", fmt(core["min_ess_bulk"].min(), 1)))
        lines.append(macro("CoreMaxRhat", fmt(core["max_rhat"].max(), 5)))
        lines.append(macro("CoreFits", int(core.groupby(["model", "fold"]).ngroups)))

    values = {}
    for line in lines:
        match = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line)
        if not match:
            raise ValueError(f"unparsable macro line: {line}")
        values[match.group(1)] = match.group(2)
    prefix = latex_name(selected)
    for key in list(values):
        if key.startswith(prefix):
            alias = f"Sel{key[len(prefix):]}"
            if alias in values and values[alias] != values[key]:
                raise ValueError(f"alias collision for {alias}")
            values[alias] = values[key]
    return [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in values.items()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="formal_20260930")
    args = parser.parse_args(argv)
    GENERATED.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(WORK_ROOT / "tables" / "all_metrics.csv")
    selection = read_json(WORK_ROOT / "tables" / "selection.json")
    for name, content in build_tables(metrics, args.run_id).items():
        (GENERATED / name).write_text(content, encoding="utf-8")
        print(GENERATED / name)
    macros = build_macros(metrics, selection, args.run_id)
    header = "% generated by src/make_report_inputs.py; do not edit by hand\n"
    (GENERATED / "macros.tex").write_text(header + "\n".join(macros) + "\n", encoding="utf-8")
    print(GENERATED / "macros.tex", len(macros), "macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
