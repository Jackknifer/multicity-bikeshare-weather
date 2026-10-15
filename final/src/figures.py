import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import WORK_ROOT, read_json
from model import load_model_config, resolve_spec
from model import flat_posterior, select_samples

FIGURE_DIR = WORK_ROOT / "figures"
TABLE_DIR = WORK_ROOT / "tables"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "#221815",
    "axes.labelcolor": "#221815",
    "xtick.color": "#221815",
    "ytick.color": "#221815",
})

PURPLE = "#750F6D"
GOLD = "#DDA300"
INK = "#221815"
GREY = "#6E6A67"
BLUE = "#2F6F9F"
GREEN = "#2E7D5B"
RED = "#B5341F"


def save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{name}.pdf"
    png = FIGURE_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return pdf, png


def figure_city_overview():
    audit = pd.read_csv(WORK_ROOT / "data" / "city_data_audit.csv")
    audit = audit.sort_values("usage_median", ascending=False).reset_index(drop=True)
    audit["first"] = pd.to_datetime(audit["first_date"])
    audit["last"] = pd.to_datetime(audit["last_date"])
    audit["scope"] = np.where(
        audit["in_prediction_23"] == 1, "prediction 23",
        np.where(audit["in_primary_25"] == 1, "primary only", "excluded"),
    )
    colors = {"prediction 23": PURPLE, "primary only": GOLD, "excluded": GREY}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 6.4), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
    order = audit.iloc[::-1]
    positions = np.arange(len(order))
    axes[0].barh(positions, order["usage_median"], color=[colors[s] for s in order["scope"]], height=0.7)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(order["city"], fontsize=6.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("median daily usage (log scale)")
    axes[0].set_title("(a) system size")
    for value, position in zip(order["usage_median"], positions):
        if value == 0:
            axes[0].text(0.15, position, "0", va="center", fontsize=5.5, color=RED)

    axes[1].barh(positions, order["zero_usage_share"] * 100.0,
                 color=[colors[s] for s in order["scope"]], height=0.7)
    axes[1].set_yticks([])
    axes[1].set_xlabel("zero-usage days (%)")
    axes[1].set_title("(b) inactive days")

    for position, (_, row) in zip(positions, order.iterrows()):
        axes[2].plot([row["first"], row["last"]], [position, position],
                     color=colors[row["scope"]], linewidth=2.2, solid_capstyle="butt")
    axes[2].set_yticks([])
    axes[2].set_xlabel("study period")
    axes[2].set_title("(c) observation window")
    axes[2].axvline(pd.Timestamp("2017-07-16"), color=GREY, linestyle=":", linewidth=0.9)
    axes[2].axvline(pd.Timestamp("2017-11-05"), color=GREY, linestyle=":", linewidth=0.9)
    handles = [plt.Line2D([0], [0], color=colors[key], linewidth=4, label=key)
               for key in ("prediction 23", "primary only", "excluded")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=7)
    fig.suptitle("City coverage, system size and inactivity in the frozen panel (29 cities)", y=0.98)
    return save(fig, "fig01_city_overview")


def _effect_city_frame(path):
    payload = read_json(path)
    frame = pd.DataFrame(payload["cities"])
    frame["model"] = payload["model"]
    frame["fold"] = payload["fold"]
    return frame


def figure_rain_forest(effect_paths, output_name="fig02_rain_forest"):
    frames = [_effect_city_frame(path) for path in effect_paths]
    combined = pd.concat(frames, ignore_index=True)
    models = list(dict.fromkeys(combined["model"]))
    cities = sorted(combined["city"].unique())
    height = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 8.6))
    palette = [PURPLE, GOLD, BLUE, GREEN]
    for offset, model in enumerate(models):
        block = combined.loc[combined["model"] == model].set_index("city").reindex(cities)
        positions = np.arange(len(cities)) + (offset - (len(models) - 1) / 2) * height
        median = block["rain_10mm_percent"].apply(lambda item: item["median"]).to_numpy()
        low = block["rain_10mm_percent"].apply(lambda item: item["q025"]).to_numpy()
        high = block["rain_10mm_percent"].apply(lambda item: item["q975"]).to_numpy()
        ax.errorbar(median, positions, xerr=[median - low, high - median], fmt="o",
                    markersize=3.2, linewidth=1.1, capsize=1.6,
                    color=palette[offset % len(palette)], label=model)
    ax.axvline(0.0, color=INK, linewidth=0.8)
    ax.set_yticks(np.arange(len(cities)))
    ax.set_yticklabels(cities, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("change in the conditional median of (1 + usage), %\nfor a +10 mm step from the city median daily precipitation")
    ax.set_title("City-specific precipitation association with 95% posterior intervals")
    ax.legend(loc="lower left", fontsize=8)
    return save(fig, output_name)


def _temperature_grid(mean_c, low_c, high_c, points=121):
    span = high_c - low_c
    return np.linspace(low_c - 0.25 * span, high_c + 0.25 * span, points)


def figure_temperature_curves(run_id, model, fold, cities, output_name="fig03_temperature_curves"):
    directory = WORK_ROOT / "runs" / run_id / model / fold
    config = load_model_config()
    spec = resolve_spec(config, model)
    prep = read_json(directory / "preprocessing.json")
    import arviz as az

    idata = az.from_netcdf(directory / "posterior.nc")
    take = select_samples(idata, 2000, 27)
    names = prep["city_names"]
    beta_temp = flat_posterior(idata, "beta_temp_city")[take]
    beta_temp_sq = flat_posterior(idata, "beta_temp_sq_city")[take]
    observed = pd.read_csv(WORK_ROOT / "data" / "city_data_audit.csv").set_index("city")

    fig, axes = plt.subplots(1, len(cities), figsize=(3.0 * len(cities), 3.4), sharey=True)
    if len(cities) == 1:
        axes = [axes]
    for ax, city in zip(axes, cities):
        index = names.index(city)
        mean_c = prep["city_temp_mean_c"][city]
        square_center = prep["temp_sq_center"][city]
        low_c = observed.loc[city, "utci_mean_c_min"]
        high_c = observed.loc[city, "utci_mean_c_max"]
        grid = _temperature_grid(mean_c, low_c, high_c)
        scaled = (grid - mean_c) / 10.0
        delta = (beta_temp[:, index][:, None] * scaled[None, :]
                 + beta_temp_sq[:, index][:, None] * (scaled[None, :] ** 2 - square_center))
        relative = 100.0 * np.expm1(delta)
        median = np.median(relative, axis=0)
        ax.fill_between(grid, np.quantile(relative, 0.025, axis=0),
                        np.quantile(relative, 0.975, axis=0), color=PURPLE, alpha=0.16)
        ax.plot(grid, median, color=PURPLE, linewidth=1.8)
        inside = (grid >= low_c) & (grid <= high_c)
        ax.plot(grid[~inside], median[~inside], color=PURPLE, linewidth=1.8,
                linestyle=":", alpha=0.75)
        ax.axvspan(low_c, high_c, color=GOLD, alpha=0.10)
        ax.axvline(low_c, color=GOLD, linewidth=0.8)
        ax.axvline(high_c, color=GOLD, linewidth=0.8)
        ax.axhline(0.0, color=INK, linewidth=0.7)
        ax.set_title(f"{city}\nobserved {low_c:.0f} to {high_c:.0f} $^\\circ$C", fontsize=8.5)
        ax.set_xlabel("daily mean UTCI ($^\\circ$C)")
    axes[0].set_ylabel("change in the conditional median\nof (1 + usage), %")
    fig.suptitle(f"Temperature association relative to each city's training-period mean ({model}, {fold})", y=1.02)
    fig.text(0.5, -0.06,
             "shaded band: 95% posterior interval; gold region: observed training UTCI range; dotted curve: extrapolation",
             ha="center", fontsize=7.5, color=GREY)
    return save(fig, output_name)


def figure_residual_acf(run_id, entries, output_name="fig04_residual_acf"):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for label, model, fold, colour in entries:
        path = WORK_ROOT / "runs" / run_id / model / fold / "residual_acf.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing residual ACF file: {path}")
        frame = pd.read_csv(path)
        for lag, block in frame.groupby("lag"):
            axes[0].plot(np.full(len(block), lag) + np.linspace(-0.18, 0.18, len(block)),
                         block["autocorrelation"], marker="o", markersize=1.6,
                         linestyle="none", color=colour, alpha=0.35)
        summary = frame.groupby("lag")["autocorrelation"].agg(["median", "min", "max"]).reset_index()
        axes[0].plot(summary["lag"], summary["median"], color=colour, linewidth=1.8,
                     marker="o", markersize=3.5, label=label)
        axes[0].fill_between(summary["lag"], summary["min"], summary["max"], color=colour, alpha=0.12)
        axes[1].hist(frame.loc[frame["lag"] == 1, "autocorrelation"], bins=14, alpha=0.45,
                     color=colour, label=label)
    axes[0].axhline(0.0, color=INK, linewidth=0.8)
    axes[0].set_xlabel("lag (days)")
    axes[0].set_ylabel("residual autocorrelation")
    axes[0].set_title("(a) lag profile per city")
    axes[0].legend(fontsize=7.5)
    axes[1].set_xlabel("lag-1 autocorrelation")
    axes[1].set_ylabel("cities")
    axes[1].set_title("(b) city distribution at lag 1")
    axes[1].legend(fontsize=7.5)
    fig.suptitle("Training residual autocorrelation on contiguous day pairs", y=1.03)
    return save(fig, output_name)


def figure_forecast_scores(run_id, models, output_name="fig05_forecast_scores"):
    metrics = pd.read_csv(TABLE_DIR / "all_metrics.csv")
    metrics = metrics.loc[metrics["model"].isin(models)]
    folds = [fold for fold in ("dev1", "dev2", "final") if fold in set(metrics["fold"])]
    tasks = ["block", "one_step"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    width = 0.8 / len(models)
    palette = [PURPLE, GOLD, BLUE, GREEN, RED]
    for ax, column, title, ylabel in (
        (axes[0], "city_equal_wis_log", "(a) weighted interval score", "city-equal WIS (lower is better)"),
        (axes[1], "city_equal_mean_log_predictive_density", "(b) log predictive density",
         "city-equal mean log density (higher is better)"),
        (axes[2], "city_equal_coverage_90", "(c) 90% interval coverage", "city-equal coverage"),
    ):
        positions = np.arange(len(folds) * len(tasks))
        labels = []
        for index, (fold, task) in enumerate([(f, t) for f in folds for t in tasks]):
            labels.append(f"{fold}\n{task}")
            block = metrics.loc[(metrics["fold"] == fold) & (metrics["task"] == task)]
            for offset, model in enumerate(models):
                row = block.loc[block["model"] == model]
                if row.empty:
                    continue
                value = float(row[column].iloc[0])
                ax.bar(index + (offset - (len(models) - 1) / 2) * width, value, width=width,
                       color=palette[offset % len(palette)], label=model if index == 0 else None)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title)
        ax.set_ylabel(ylabel, fontsize=8)
        if column == "city_equal_coverage_90":
            ax.axhline(0.90, color=RED, linewidth=1.0, linestyle="--")
            ax.text(0.02, 0.92, "nominal 90%", transform=ax.transAxes, fontsize=7, color=RED)
    axes[0].legend(fontsize=7.5, ncol=2)
    fig.suptitle("Predictive skill by fold and task, city-equal aggregates", y=1.03)
    return save(fig, output_name)


def figure_city_trajectories(run_id, model, fold, cities, output_name="fig06_city_trajectories"):
    directory = WORK_ROOT / "runs" / run_id / model / fold
    predictions = pd.read_csv(directory / "predictions.csv", parse_dates=["date"])
    predictions = predictions.loc[predictions["task"] == "block"]
    fig, axes = plt.subplots(len(cities), 1, figsize=(9.0, 2.5 * len(cities)), sharex=False)
    if len(cities) == 1:
        axes = [axes]
    for ax, city in zip(axes, cities):
        block = predictions.loc[predictions["city"] == city].sort_values("date")
        ax.fill_between(block["date"], block["lower_95"], block["upper_95"], color=PURPLE, alpha=0.12,
                        label="95% prediction interval")
        ax.fill_between(block["date"], block["lower_50"], block["upper_50"], color=PURPLE, alpha=0.24,
                        label="50% prediction interval")
        ax.plot(block["date"], block["median_log"], color=PURPLE, linewidth=1.6, label="median forecast")
        ax.plot(block["date"], block["observed_log"], color=INK, linewidth=1.3, label="observed")
        covered = float(block["covered_90"].mean())
        ax.set_title(f"{city}: 90% coverage {covered:.2f}, block WIS "
                     f"{block['wis_log'].mean():.3f}", fontsize=9)
        ax.set_ylabel("log(1 + usage)")
        ax.legend(fontsize=7, ncol=4, loc="upper left")
    axes[-1].set_xlabel("date")
    fig.suptitle(f"56-day block forecasts with actual weather ({model}, {fold})", y=1.0)
    fig.tight_layout()
    return save(fig, output_name)


def figure_pooling_ablation(run_id, fold, output_name="fig07_pooling_ablation"):
    frames = []
    for model in ("A0", "A1"):
        directory = WORK_ROOT / "runs" / run_id / model / fold
        if not (directory / "posterior_summary.csv").exists():
            raise FileNotFoundError(f"missing posterior summary for {model}")
        summary = pd.read_csv(directory / "posterior_summary.csv")
        block = summary.loc[summary["parameter"].str.startswith("beta_rain_city[")].copy()
        block["city"] = block["parameter"].str.extract(r"\[(.*)\]")
        block["model"] = model
        frames.append(block)
    combined = pd.concat(frames, ignore_index=True)
    cities = sorted(combined["city"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 6.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
    positions = np.arange(len(cities))
    palette = {"A0": GREY, "A1": PURPLE}
    for model in ("A0", "A1"):
        block = combined.loc[combined["model"] == model].set_index("city").reindex(cities)
        axes[0].errorbar(block["mean"], positions + (0.16 if model == "A1" else -0.16),
                         xerr=1.96 * block["sd"], fmt="o", markersize=3.4, linewidth=1.1,
                         capsize=1.6, color=palette[model], label=f"{model} 95% credible interval")
        axes[1].plot(block["sd"], positions + (0.16 if model == "A1" else -0.16), "o",
                     markersize=3.4, color=palette[model], label=model)
    axes[0].axvline(0.0, color=INK, linewidth=0.8)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(cities, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("posterior mean of the city precipitation coefficient")
    axes[0].set_title("(a) coefficient location")
    axes[0].legend(fontsize=7.5, loc="lower left")
    axes[1].set_yticks([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("posterior standard deviation")
    axes[1].set_title("(b) uncertainty")
    axes[1].legend(fontsize=7.5)
    fig.suptitle(f"Weather-coefficient pooling ablation on {fold}: no pooling (A0) versus partial pooling (A1)", y=0.98)
    return save(fig, output_name)


def figure_prior_predictive(run_id, model, fold, output_name="figA1_prior_predictive"):
    path = WORK_ROOT / "runs" / run_id / model / fold / "prior_predictive.json"
    payload = read_json(path)
    panels = [
        ("intercept_usage_quantiles", "implied usage scale", True),
        ("rain_10mm_percent_quantiles", "rain +10 mm, %", False),
        ("temp_10c_percent_quantiles", "UTCI +10 $^\\circ$C, %", False),
        ("trend_year_log_change_quantiles", "trend over one year, log units", False),
        ("phi_quantiles", "AR(1) persistence", False),
    ]
    panels = [item for item in panels if item[0] in payload]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels), 3.0))
    quantiles = sorted(float(key) for key in payload[panels[0][0]])
    for ax, (key, title, log_scale) in zip(axes, panels):
        values = [payload[key][str(q)] for q in quantiles]
        ax.plot(quantiles, values, marker="o", markersize=3, color=PURPLE, linewidth=1.4)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("prior quantile", fontsize=8)
        if log_scale:
            ax.set_yscale("log")
        ax.tick_params(labelsize=7)
    if "observed_log_usage_range" in payload:
        low, high = payload["observed_log_usage_range"]
        axes[0].axhline(np.expm1(high), color=RED, linewidth=0.9, linestyle="--")
        axes[0].axhline(np.expm1(low), color=RED, linewidth=0.9, linestyle="--")
        axes[0].text(0.02, 0.96, "observed usage range", transform=axes[0].transAxes,
                     fontsize=6.5, color=RED, va="top")
    fig.suptitle(f"Prior predictive quantiles for {model} on {fold}", y=1.03)
    return save(fig, output_name)


def figure_all_city_curves(run_id, model, fold, output_name="figA2_all_city_curves"):
    import arviz as az

    directory = WORK_ROOT / "runs" / run_id / model / fold
    prep = read_json(directory / "preprocessing.json")
    idata = az.from_netcdf(directory / "posterior.nc")
    take = select_samples(idata, 1500, 27)
    names = prep["city_names"]
    beta_temp = flat_posterior(idata, "beta_temp_city")[take]
    beta_temp_sq = flat_posterior(idata, "beta_temp_sq_city")[take]
    observed = pd.read_csv(WORK_ROOT / "data" / "city_data_audit.csv").set_index("city")

    columns = 5
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 1.9 * rows), sharex=False)
    axes = np.atleast_2d(axes)
    for index, city in enumerate(names):
        ax = axes[index // columns][index % columns]
        mean_c = prep["city_temp_mean_c"][city]
        square_center = prep["temp_sq_center"][city]
        low_c = observed.loc[city, "utci_mean_c_min"]
        high_c = observed.loc[city, "utci_mean_c_max"]
        grid = np.linspace(low_c, high_c, 60)
        scaled = (grid - mean_c) / 10.0
        delta = (beta_temp[:, index][:, None] * scaled[None, :]
                 + beta_temp_sq[:, index][:, None] * (scaled[None, :] ** 2 - square_center))
        relative = 100.0 * np.expm1(delta)
        ax.fill_between(grid, np.quantile(relative, 0.025, axis=0),
                        np.quantile(relative, 0.975, axis=0), color=PURPLE, alpha=0.18)
        ax.plot(grid, np.median(relative, axis=0), color=PURPLE, linewidth=1.1)
        ax.axhline(0.0, color=INK, linewidth=0.6)
        ax.set_title(city, fontsize=7)
        ax.tick_params(labelsize=6)
    for index in range(len(names), rows * columns):
        axes[index // columns][index % columns].axis("off")
    fig.suptitle(f"Temperature association inside the observed UTCI range ({model}, {fold})", y=1.0)
    fig.tight_layout()
    return save(fig, output_name)


def figure_convergence(run_id, entries, output_name="figA3_convergence"):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    labels = []
    for index, (label, model, fold) in enumerate(entries):
        summary = pd.read_csv(WORK_ROOT / "runs" / run_id / model / fold / "posterior_summary.csv")
        labels.append(label)
        axes[0].boxplot(summary["r_hat"].to_numpy(), positions=[index], widths=0.55,
                        showfliers=True, patch_artist=True,
                        boxprops={"facecolor": PURPLE, "alpha": 0.35},
                        medianprops={"color": PURPLE})
        axes[1].boxplot(np.log10(summary["ess_bulk"].to_numpy()), positions=[index], widths=0.55,
                        showfliers=True, patch_artist=True,
                        boxprops={"facecolor": GOLD, "alpha": 0.45},
                        medianprops={"color": GOLD})
    axes[0].axhline(1.01, color=RED, linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("rank-normalised $\\hat{R}$")
    axes[0].set_title("(a) $\\hat{R}$ across all parameters")
    axes[1].axhline(np.log10(400), color=RED, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("$\\log_{10}$ bulk ESS")
    axes[1].set_title("(b) bulk effective sample size")
    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
    fig.suptitle("Convergence diagnostics for every formal fit", y=1.02)
    return save(fig, output_name)


def figure_sensitivity(run_id, selected, fold, output_name="figA4_sensitivity"):
    entries = [(f"{selected} prior x1", selected), (f"{selected} prior x0.5", f"{selected}_s05"),
               (f"{selected} prior x2", f"{selected}_s20")]
    rows = []
    for label, name in entries:
        path = WORK_ROOT / "runs" / run_id / name / fold / "effects.json"
        if not path.exists():
            raise FileNotFoundError(f"missing effects for {name}")
        payload = read_json(path)
        rows.append({
            "label": label,
            "rain": payload["rain_10mm_city_equal_percent"],
            "temp": payload["temp_10c_city_equal_percent"],
        })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for ax, key, title in ((axes[0], "rain", "precipitation +10 mm"),
                           (axes[1], "temp", "UTCI +10 $^\\circ$C")):
        positions = np.arange(len(rows))
        median = np.array([row[key]["median"] for row in rows])
        low = np.array([row[key]["q025"] for row in rows])
        high = np.array([row[key]["q975"] for row in rows])
        ax.errorbar(median, positions, xerr=[median - low, high - median], fmt="o",
                    markersize=5, linewidth=1.4, capsize=3, color=PURPLE)
        ax.axvline(0.0, color=INK, linewidth=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels([row["label"] for row in rows], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("change in the conditional median of $(1+\\text{usage})$, %")
    fig.suptitle(f"Weather prior-scale sensitivity on {fold}", y=1.02)
    return save(fig, output_name)


def figure_posterior_predictive(run_id, model, fold, output_name="figA5_posterior_predictive"):
    frame = pd.read_csv(WORK_ROOT / "runs" / run_id / model / fold / "ppc_train.csv")
    per_city = frame.groupby("city").agg(
        coverage=("covered90", "mean"),
        rmse=("residual_log", lambda values: float(np.sqrt(np.mean(values ** 2)))),
    ).sort_values("coverage")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    positions = np.arange(len(per_city))
    axes[0].barh(positions, per_city["coverage"] * 100.0, color=PURPLE, height=0.7)
    axes[0].axvline(90.0, color=RED, linestyle="--", linewidth=1.0)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(per_city.index, fontsize=7)
    axes[0].set_xlabel("training 90% posterior predictive coverage (%)")
    axes[0].set_title("(a) in-sample coverage")
    axes[1].barh(positions, per_city["rmse"], color=GOLD, height=0.7)
    axes[1].set_yticks([])
    axes[1].set_xlabel("training log-residual RMSE")
    axes[1].set_title("(b) residual scale")
    fig.suptitle(f"Posterior predictive check by city ({model}, {fold})", y=1.0)
    return save(fig, output_name)


def figure_prediction_attribution(run_id, output_name="fig08_prediction_attribution"):
    metrics = pd.read_csv(TABLE_DIR / "all_metrics.csv")
    labels = [
        ("M0", "weekend\n+ harmonic"),
        ("M1nt", "weekday\n+ harmonic"),
        ("M1", "weekday\n+ trend\n+ harmonic"),
        ("M2", "weekday\n+ trend\n+ harmonic\n+ AR(1)"),
    ]
    folds = ["dev1", "dev2", "final"]
    fig, axes = plt.subplots(1, len(folds), figsize=(4.0 * len(folds), 3.8), sharey=True)
    palette = [GREY, GOLD, RED, PURPLE]
    for ax, fold in zip(axes, folds):
        positions = []
        heights = []
        colours = []
        ticks = []
        for index, (model, _) in enumerate(labels):
            block = metrics.loc[(metrics["model"] == model) & (metrics["fold"] == fold)
                                & (metrics["task"] == "block")]
            if block.empty:
                continue
            positions.append(index)
            heights.append(float(block.iloc[0]["city_equal_wis_log"]))
            colours.append(palette[index])
            ticks.append(index)
        ax.bar(positions, heights, color=colours, width=0.68)
        for position, height in zip(positions, heights):
            ax.text(position, height + 0.006, f"{height:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([text for _, text in labels], fontsize=7)
        ax.set_title(fold)
        if fold == "final":
            ax.set_facecolor("#FBF7F2")
    axes[0].set_ylabel("city-equal WIS, block task\n(lower is better)")
    fig.suptitle("Which added term breaks the 56-day forecast? City linear trend", y=1.03)
    fig.text(0.5, -0.04,
             "M1nt is a post-hoc diagnostic specification used only to attribute the final-fold degradation; "
             "it is not a selection candidate.",
             ha="center", fontsize=7.5, color=GREY)
    return save(fig, output_name)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="formal")
    parser.add_argument("--only", default=None)
    args = parser.parse_args(argv)
    produced = []
    produced.append(figure_city_overview())
    for item in produced:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
