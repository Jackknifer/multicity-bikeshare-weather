from pathlib import Path
import hashlib
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "clean" / "bikeshare_weather_daily.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
EXPECTED_SHA256 = "6e30157ee8061f76142b95dd9ca7615bdea068f57c2fbd82796c7ac5e62af33e"

PRIMARY_ACTIVE_SHARE = 0.99
UTCI_EDGES = np.round(np.arange(-4.0, 3.0001, 0.5), 3)
UTCI_ABS_EDGES = np.arange(-35.0, 40.0, 5.0)
RAIN_EDGES = [-0.001, 0.001, 1.0, 3.0, 8.0, 20.0, 1e9]
RAIN_LABELS = ["0", "0-1", "1-3", "3-8", "8-20", ">20"]
SEASON_EDGES = np.arange(0.5, 366.5, 14.0)

C_PRIMARY = "#2F4B7C"
C_WARM = "#B4472F"
C_COOL = "#3E8E6E"
C_NEUTRAL = "#5A5A5A"
C_GRID = "#D8D8D8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

FULL_W = 5.5
HALF_W = 2.65


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(fig, stem, png=True):
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    if png:
        fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def load():
    require(sha256(DATA_PATH) == EXPECTED_SHA256, "daily data checksum mismatch")
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    require(len(data) == 10578, "daily row count mismatch")
    require(data["city"].nunique() == 29, "daily city count mismatch")
    require(not data.duplicated(["city", "date"]).any(), "duplicate city-date keys")
    require(data.isna().sum().sum() == 0, "daily data contain missing cells")
    return data


def city_table(data):
    """每个城市的样本规模、活跃度与天气背景。"""
    records = []
    for city, group in data.groupby("city", sort=True):
        usage_median = float(group["usage"].median())
        records.append({
            "city": city,
            "country": group["country"].iloc[0],
            "climate_code": group["climate_code"].iloc[0],
            "days": int(len(group)),
            "zero_usage_days": int((group["usage"] == 0).sum()),
            "active_day_share": float(group["active_day_share"].iloc[0]),
            "continuous_system": int(group["continuous_system"].iloc[0]),
            "usage_median": usage_median,
            "usage_mean": float(group["usage"].mean()),
            "usage_log1p_mean": float(group["usage_log1p"].mean()),
            "utci_mean_c": float(group["utci_mean_c"].mean()),
            "utci_min_c": float(group["utci_min_c"].min()),
            "utci_max_c": float(group["utci_max_c"].max()),
            "rain_mean_mm": float(group["precipitation_mm"].mean()),
            "rain_zero_share": float((group["precipitation_mm"] == 0).mean()),
        })
    table = pd.DataFrame(records).set_index("city")
    require(len(table) == 29, "city table must cover 29 cities")
    return table


def city_centered(frame):
    """把 log1p 使用量减去城市均值，去掉城市规模差异。"""
    centered = frame["usage_log1p"] - frame.groupby("city")["usage_log1p"].transform("mean")
    return centered.to_numpy()


def binned_by_city(frame, x_values, y_values, edges):
    """按 x 分箱，返回每箱的城市均值、城市间分位数与样本量。"""
    frame = frame.assign(_x=x_values, _y=y_values)
    frame = frame.assign(_bin=np.digitize(frame["_x"].to_numpy(), edges) - 1)
    frame = frame.loc[(frame["_bin"] >= 0) & (frame["_bin"] < len(edges) - 1)]
    per_city = frame.groupby(["city", "_bin"])["_y"].mean().unstack("_bin")
    per_city = per_city.reindex(columns=range(len(edges) - 1))
    counts = frame.groupby("_bin").size().reindex(range(len(edges) - 1)).fillna(0)
    curve = pd.DataFrame({
        "bin": np.arange(len(edges) - 1),
        "center": (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2.0,
        "mean": per_city.mean(axis=0).to_numpy(),
        "q25": per_city.quantile(0.25, axis=0).to_numpy(),
        "q75": per_city.quantile(0.75, axis=0).to_numpy(),
        "cities": per_city.notna().sum(axis=0).to_numpy(),
        "observations": counts.to_numpy(),
    })
    return curve, per_city


def fig_data_overview(table):
    primary = table.loc[table["continuous_system"] == 1].sort_values("usage_median")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.0),
                             gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.30})

    ax = axes[0]
    ypos = np.arange(len(primary))
    ax.barh(ypos, primary["usage_median"], height=0.72, color=C_PRIMARY, alpha=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(primary.index, fontsize=5.2)
    ax.set_xscale("log")
    ax.set_xlabel("Median daily estimated hires (log scale)")
    ax.set_title("(a) Primary sample: 25 cities")
    ax.set_xlim(5, 2.2e5)
    ax.grid(axis="y", visible=False)
    ax.annotate(f"{primary['usage_median'].iloc[0]:.1f}",
                xy=(primary["usage_median"].iloc[0], 0), xytext=(6.5, 0.6),
                fontsize=5.4, color="#333333")
    ax.annotate(f"{primary['usage_median'].iloc[-1]:,.0f}",
                xy=(primary["usage_median"].iloc[-1], len(primary) - 1),
                xytext=(primary["usage_median"].iloc[-1] * 1.25, len(primary) - 1.8),
                fontsize=5.4, color="#333333")
    spread = primary["usage_median"].iloc[-1] / primary["usage_median"].iloc[0]
    ax.text(0.02, 0.965, f"{spread:,.0f}$\\times$ spread across the primary sample",
            transform=ax.transAxes, fontsize=5.8, va="top", color="#33405C",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#F2F5FA",
                      edgecolor="#C3CEE0", linewidth=0.5))

    ax = axes[1]
    ordered = table.sort_values("active_day_share")
    colors = [C_PRIMARY if v == 1 else C_WARM for v in ordered["continuous_system"]]
    ax.barh(np.arange(len(ordered)), ordered["active_day_share"] * 100.0,
            height=0.72, color=colors, alpha=0.9)
    ax.axvline(PRIMARY_ACTIVE_SHARE * 100.0, ls="--", lw=0.9, color=C_WARM)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered.index, fontsize=4.6)
    ax.set_xlabel("Share of analysis days with positive hires (%)")
    ax.set_title("(b) Continuous-system filter, 29 cities")
    ax.set_xlim(0, 104)
    ax.grid(axis="y", visible=False)
    ax.text(PRIMARY_ACTIVE_SHARE * 100.0 - 1.5, 1.0, "99% threshold",
            rotation=90, fontsize=5.2, color=C_WARM, ha="right", va="bottom")
    handles = [Patch(facecolor=C_PRIMARY, alpha=0.9, label="Primary sample (25)"),
               Patch(facecolor=C_WARM, alpha=0.9, label="Excluded (4)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=5.6)

    save(fig, "report_fig1_data_overview")
    return primary


def fig_weather_usage(data, primary):
    centered = city_centered(primary)
    frame = primary.assign(_centered=centered)

    utci_curve, utci_city = binned_by_city(frame, primary["utci_mean_c"].to_numpy(),
                                           centered, UTCI_ABS_EDGES)
    rain_curve, rain_city = binned_by_city(frame, primary["precipitation_mm"].to_numpy(),
                                           centered, RAIN_EDGES)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 1.65),
                             gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    ok = utci_curve["cities"] >= 5
    ax.fill_between(utci_curve.loc[ok, "center"], utci_curve.loc[ok, "q25"],
                    utci_curve.loc[ok, "q75"], color=C_PRIMARY, alpha=0.18, lw=0)
    ax.plot(utci_curve.loc[ok, "center"], utci_curve.loc[ok, "mean"],
            color=C_PRIMARY, lw=1.3, marker="o", ms=2.6)
    ax.axhline(0.0, color=C_NEUTRAL, lw=0.6, ls=":")
    ax.set_xlabel("Daily mean UTCI ($^\\circ$C)")
    ax.set_ylabel("Within-city centred $\\log(1+$hires$)$")
    ax.set_title("(a) Thermal response, 25 cities")
    populated = utci_curve.loc[ok].dropna(subset=["mean"])
    peak = populated.sort_values("mean").iloc[-1]
    ax.annotate(f"peak near {peak['center']:.0f} $^\\circ$C",
                xy=(peak["center"], peak["mean"]),
                xytext=(populated["center"].min() + 1.0, peak["mean"] + 0.06),
                fontsize=5.8, color=C_WARM,
                arrowprops=dict(arrowstyle="->", color=C_WARM, lw=0.6))
    ax.text(0.025, 0.05, "band: 25th-75th percentile across cities",
            transform=ax.transAxes, fontsize=5.4, color=C_NEUTRAL)

    ax = axes[1]
    ok = rain_curve["cities"] >= 10
    xpos = np.arange(len(rain_curve))
    ax.fill_between(xpos[ok], rain_curve.loc[ok, "q25"], rain_curve.loc[ok, "q75"],
                    color=C_COOL, alpha=0.18, lw=0)
    ax.plot(xpos[ok], rain_curve.loc[ok, "mean"], color=C_COOL, lw=1.3,
            marker="o", ms=2.6)
    ax.axhline(0.0, color=C_NEUTRAL, lw=0.6, ls=":")
    ax.set_xticks(xpos)
    ax.set_xticklabels(RAIN_LABELS, fontsize=6.4)
    ax.set_xlabel("Daily precipitation (mm)")
    ax.set_ylabel("Within-city centred $\\log(1+$hires$)$")
    ax.set_title("(b) Rainfall response, 25 cities")
    zero = float(rain_curve.loc[rain_curve["bin"] == 0, "mean"].iloc[0])
    heavy = float(rain_curve.loc[rain_curve["bin"] == len(RAIN_LABELS) - 1, "mean"].iloc[0])
    ax.annotate(f"{np.expm1(heavy) * 100:.0f}% at the heaviest bin",
                xy=(len(RAIN_LABELS) - 1, heavy),
                xytext=(len(RAIN_LABELS) - 3.3, heavy - 0.16),
                fontsize=5.8, color=C_COOL,
                arrowprops=dict(arrowstyle="->", color=C_COOL, lw=0.6))
    ax.text(0.025, 0.05, f"zero-rain bin sits {zero:+.3f} above the city mean",
            transform=ax.transAxes, fontsize=5.4, color=C_NEUTRAL)

    save(fig, "report_fig2_weather_usage")
    return utci_curve, rain_curve, utci_city, rain_city


def fig_utci_anomaly_curve(primary):
    centered = city_centered(primary)
    frame = primary.assign(_centered=centered)
    curve, _ = binned_by_city(frame, primary["utci_anomaly_10c"].to_numpy(),
                              centered, UTCI_EDGES)
    ok = curve["cities"] >= 10
    fig, ax = plt.subplots(figsize=(FULL_W, 2.3))
    ax.fill_between(curve.loc[ok, "center"], curve.loc[ok, "q25"], curve.loc[ok, "q75"],
                    color=C_PRIMARY, alpha=0.18, lw=0)
    ax.plot(curve.loc[ok, "center"], curve.loc[ok, "mean"], color=C_PRIMARY,
            lw=1.3, marker="o", ms=2.6)
    ax.axhline(0.0, color=C_NEUTRAL, lw=0.6, ls=":")
    ax.set_xlabel("UTCI anomaly from the city mean (10 $^\\circ$C units)")
    ax.set_ylabel("Within-city centred $\\log(1+$hires$)$")
    ax.set_title("Thermal response on the model scale: anomaly from the city mean")
    for _, row in curve.loc[ok].iterrows():
        ax.text(row["center"], row["mean"] - 0.055, f"{int(row['observations'])}",
                fontsize=4.2, ha="center", color=C_NEUTRAL)
    ax.text(0.025, 0.06, "numbers give city-days per bin",
            transform=ax.transAxes, fontsize=5.2, color=C_NEUTRAL)
    save(fig, "figS7_utci_anomaly_curve")
    return curve


def fig_seasonality(data, primary):
    centered = city_centered(primary)
    frame = primary.assign(_centered=centered)
    season_curve, season_city = binned_by_city(
        frame, primary["day_of_year"].to_numpy(), centered, SEASON_EDGES)

    weekend = (primary.groupby(["city", "weekend"])["usage_log1p"].mean()
               .unstack("weekend"))
    weekend["difference"] = weekend[1] - weekend[0]
    weekend = weekend.sort_values("difference")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 1.3),
                             gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.30})

    ax = axes[0]
    ok = season_curve["cities"] >= 10
    ax.fill_between(season_curve.loc[ok, "center"], season_curve.loc[ok, "q25"],
                    season_curve.loc[ok, "q75"], color=C_PRIMARY, alpha=0.18, lw=0)
    ax.plot(season_curve.loc[ok, "center"], season_curve.loc[ok, "mean"],
            color=C_PRIMARY, lw=1.3)
    ax.axhline(0.0, color=C_NEUTRAL, lw=0.6, ls=":")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Within-city centred $\\log(1+$hires$)$")
    ax.set_title("(a) Seasonal profile, 25 cities")
    ax.set_xlim(0, 366)
    ax.set_xticks([1, 91, 182, 274, 365])
    ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"], fontsize=6.4)

    ax = axes[1]
    ypos = np.arange(len(weekend))
    ax.barh(ypos, weekend["difference"], height=0.72, color=C_WARM, alpha=0.88)
    ax.axvline(0.0, color=C_NEUTRAL, lw=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(weekend.index, fontsize=4.6)
    ax.set_xlabel("Weekend minus weekday, $\\log(1+$hires$)$")
    ax.set_title("(b) Weekend effect by city")
    ax.grid(axis="y", visible=False)
    n_negative = int((weekend["difference"] < 0).sum())
    ax.text(0.97, 0.06, f"{n_negative} of {len(weekend)} cities negative",
            transform=ax.transAxes, fontsize=5.8, ha="right", color="#33405C",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#F2F5FA",
                      edgecolor="#C3CEE0", linewidth=0.5))

    save(fig, "report_fig3_seasonality")
    return season_curve, weekend


def fig_usage_distribution(data, primary):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2),
                             gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    positive = primary.loc[primary["usage"] > 0, "usage"]
    ax.hist(positive, bins=np.logspace(0, np.log10(positive.max()), 40),
            color=C_PRIMARY, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Daily estimated hires (log scale)")
    ax.set_ylabel("City-days")
    ax.set_title("(a) Daily hires, positive days only")

    ax = axes[1]
    ax.hist(primary["usage_log1p"], bins=40, color=C_COOL, alpha=0.9)
    ax.set_xlabel("$\\log(1+$daily hires$)$")
    ax.set_ylabel("City-days")
    ax.set_title("(b) Model response scale")
    ax.axvline(0.0, color=C_NEUTRAL, lw=0.6, ls=":")

    save(fig, "figS1_usage_distribution")


def fig_weather_distribution(table):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.35),
                             gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    ordered = table.sort_values("utci_mean_c")
    ypos = np.arange(len(ordered))
    for k, (city, row) in enumerate(ordered.iterrows()):
        is_primary = row["continuous_system"] == 1
        color = C_PRIMARY if is_primary else C_WARM
        ax.plot([row["utci_min_c"], row["utci_max_c"]], [k, k],
                color=color, lw=1.4, alpha=0.55 if is_primary else 0.9, solid_capstyle="butt")
        ax.plot([row["utci_mean_c"]], [k], marker="o", ms=2.4,
                color=color, alpha=0.9 if is_primary else 1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ordered.index, fontsize=4.4)
    ax.set_xlabel("Daily mean UTCI ($^\\circ$C)")
    ax.set_title("(a) Thermal range by city")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ax.barh(ypos, ordered["rain_zero_share"] * 100.0, height=0.72, color=C_COOL, alpha=0.88)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ordered.index, fontsize=4.4)
    ax.set_xlabel("Days with zero precipitation (%)")
    ax.set_title("(b) Dry-day share by city")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)

    save(fig, "figS2_weather_distribution")


def fig_city_climate(table):
    primary = table.loc[table["continuous_system"] == 1]
    excluded = table.loc[table["continuous_system"] == 0]
    codes = sorted(table["climate_code"].unique())
    palette = plt.get_cmap("tab10")
    colour_of = {code: palette(i % 10) for i, code in enumerate(codes)}

    fig, ax = plt.subplots(figsize=(FULL_W, 2.75))
    size = np.sqrt(primary["usage_median"]) / 1.6 + 6.0
    ax.scatter(primary["utci_mean_c"], primary["rain_mean_mm"], s=size,
               c=[colour_of[c] for c in primary["climate_code"]],
               alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(excluded["utci_mean_c"], excluded["rain_mean_mm"], s=14,
               facecolor="none", edgecolor=C_WARM, linewidth=0.8, zorder=2)
    for city, row in primary.iterrows():
        ax.annotate(city, xy=(row["utci_mean_c"], row["rain_mean_mm"]),
                    xytext=(2.2, 2.2), textcoords="offset points",
                    fontsize=4.4, color="#333333")
    ax.set_xlabel("City-mean daily UTCI ($^\\circ$C)")
    ax.set_ylabel("City-mean daily precipitation (mm)")
    ax.set_title("Climate background of the 29 reproducible cities")
    handles = [Line2D([], [], marker="o", ls="", color=colour_of[c], ms=4.5, label=c)
               for c in codes]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none",
                          markeredgecolor=C_WARM, ms=4.5, label="excluded"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=5.0,
              ncol=2, handletextpad=0.4, columnspacing=0.8)
    ax.text(0.985, 0.03, "marker area scales with median daily hires",
            transform=ax.transAxes, fontsize=5.2, ha="right", color=C_NEUTRAL)

    save(fig, "figS3_city_climate")


def fig_city_curve_grid(primary, per_city, edges, xlabel, title, stem, labels=None):
    cities = sorted(per_city.index)
    fig, axes = plt.subplots(5, 5, figsize=(FULL_W, 5.0),
                             sharex=True, sharey=True)
    centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2.0
    for ax, city in zip(axes.ravel(), cities):
        values = per_city.loc[city].to_numpy(dtype=float)
        ok = np.isfinite(values)
        x = np.arange(len(values)) if labels is not None else centers
        ax.plot(x[ok], values[ok], color=C_PRIMARY, lw=0.9)
        ax.axhline(0.0, color=C_NEUTRAL, lw=0.45, ls=":")
        ax.set_title(city, fontsize=4.8, pad=1.5)
        ax.tick_params(labelsize=4.0, length=1.4, pad=0.8)
        ax.grid(lw=0.35, alpha=0.7)
        if labels is not None:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=3.6)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=5.4)
    for ax in axes[:, 0]:
        ax.set_ylabel("centred $\\log(1+$hires$)$", fontsize=5.0)
    fig.suptitle(title, fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, stem)


def seasonal_share_of_thermal_variance(primary):
    """每个城市的城市内 UTCI 异常值能被两个年周期项解释的比例。

    这是温度与季节项共线程度的量化：比例越高，说明天气系数越依赖季节项
    是否已在模型里，估计量的解释就越需要说清楚。
    """
    shares = {}
    for city, group in primary.groupby("city", sort=True):
        response = group["utci_anomaly_10c"].to_numpy(dtype=float)
        design = np.column_stack([
            np.ones(len(group)),
            group["season_sin"].to_numpy(dtype=float),
            group["season_cos"].to_numpy(dtype=float),
        ])
        fitted = design @ np.linalg.lstsq(design, response, rcond=None)[0]
        total = float(((response - response.mean()) ** 2).sum())
        residual = float(((response - fitted) ** 2).sum())
        shares[city] = 1.0 - residual / total
    require(all(0.0 <= value <= 1.0 for value in shares.values()),
            "seasonal variance share outside [0, 1]")
    return shares


def selected_cities(primary, count=6):
    """按城市研究期平均 UTCI 等距取代表城市，覆盖从最冷到最暖的范围。"""
    means = primary.groupby("city")["utci_mean_c"].mean().sort_values()
    targets = np.linspace(means.min(), means.max(), count)
    chosen = [min(means.index, key=lambda city: abs(means[city] - target))
              for target in targets]
    require(len(set(chosen)) == count, "representative cities must be distinct")
    return chosen


def fig_selected_city_curves(primary, per_city, edges, xlabel, title, stem,
                             cities, labels=None):
    fig, axes = plt.subplots(2, 3, figsize=(HALF_W, 2.2), sharex=True, sharey=True)
    centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2.0
    mean_utci = primary.groupby("city")["utci_mean_c"].mean()
    for ax, city in zip(axes.ravel(), cities):
        values = per_city.loc[city].to_numpy(dtype=float)
        ok = np.isfinite(values)
        x = np.arange(len(values)) if labels is not None else centers
        ax.plot(x[ok], values[ok], color=C_PRIMARY, lw=0.9, marker="o", ms=1.6)
        ax.axhline(0.0, color=C_NEUTRAL, lw=0.45, ls=":")
        ax.set_title(f"{city}  ({mean_utci[city]:.1f} $^\\circ$C mean)", fontsize=6.0, pad=1.6)
        ax.tick_params(labelsize=5.4, length=1.4, pad=0.8)
        ax.grid(lw=0.35, alpha=0.7)
        if labels is not None:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=5.0)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=6.4)
    for ax in axes[:, 0]:
        ax.set_ylabel("centred $\\log(1+$hires$)$", fontsize=6.0)
    fig.suptitle(title, fontsize=7.6, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, stem)


def fig_climate_group(primary, centered):
    frame = primary.assign(_centred=centered)
    frame = frame.assign(_bin=np.digitize(frame["utci_anomaly_10c"].to_numpy(),
                                          UTCI_EDGES) - 1)
    frame = frame.loc[(frame["_bin"] >= 0) & (frame["_bin"] < len(UTCI_EDGES) - 1)]
    centers = (UTCI_EDGES[:-1] + UTCI_EDGES[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    palette = plt.get_cmap("tab10")
    codes = sorted(frame["climate_code"].unique())
    for i, code in enumerate(codes):
        subset = frame.loc[frame["climate_code"] == code]
        n_cities = subset["city"].nunique()
        if n_cities < 2:
            continue
        curve = subset.groupby("_bin")["_centred"].mean()
        counts = subset.groupby("_bin").size()
        keep = (curve.index.isin(np.where(counts >= 20)[0]))
        ax.plot(centers[curve.index[keep]], curve[keep], lw=1.3,
                color=palette(i % 10), marker="o", ms=2.2,
                label=f"{code} ({n_cities} cities)")
    ax.axhline(0.0, color=C_NEUTRAL, lw=0.6, ls=":")
    ax.set_xlabel("UTCI anomaly from the city mean (10 $^\\circ$C units)")
    ax.set_ylabel("Within-city centred $\\log(1+$hires$)$")
    ax.set_title("Thermal response by Trewartha climate group")
    ax.legend(frameon=False, fontsize=5.4, ncol=2)
    save(fig, "figS6_climate_group")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    data = load()
    table = city_table(data)
    primary = data.loc[data["continuous_system"] == 1].copy()
    require(len(primary) == 9120, "primary sample row count mismatch")
    require(primary["city"].nunique() == 25, "primary sample city count mismatch")

    fig_data_overview(table)
    utci_curve, rain_curve, utci_city, rain_city = fig_weather_usage(data, primary)
    anomaly_curve = fig_utci_anomaly_curve(primary)
    season_curve, weekend = fig_seasonality(data, primary)
    fig_usage_distribution(data, primary)
    fig_weather_distribution(table)
    fig_city_climate(table)
    fig_city_curve_grid(primary, utci_city, UTCI_ABS_EDGES,
                        "Daily mean UTCI ($^\\circ$C)",
                        "Thermal response by city",
                        "figS4_utci_curves_grid")
    fig_city_curve_grid(primary, rain_city, RAIN_EDGES,
                        "Precipitation (mm)",
                        "Rainfall response by city",
                        "figS5_rain_curves_grid", labels=RAIN_LABELS)
    representatives = selected_cities(primary)
    fig_selected_city_curves(primary, utci_city, UTCI_ABS_EDGES,
                             "Daily mean UTCI ($^\\circ$C)",
                             "Thermal response, six representative cities",
                             "figS8_utci_curves_selected", representatives)
    fig_selected_city_curves(primary, rain_city, RAIN_EDGES,
                             "Precipitation (mm)",
                             "Rainfall response, six representative cities",
                             "figS9_rain_curves_selected", representatives,
                             labels=RAIN_LABELS)
    seasonal_share = seasonal_share_of_thermal_variance(primary)
    fig_climate_group(primary, city_centered(primary))

    primary_table = table.loc[table["continuous_system"] == 1]
    excluded_table = table.loc[table["continuous_system"] == 0]
    utci_ok = utci_curve.loc[utci_curve["cities"] >= 5].dropna(subset=["mean"])
    rain_ok = rain_curve.loc[rain_curve["cities"] >= 10].dropna(subset=["mean"])
    season_ok = season_curve.loc[season_curve["cities"] >= 10].dropna(subset=["mean"])
    anomaly_ok = anomaly_curve.loc[anomaly_curve["cities"] >= 10].dropna(subset=["mean"])
    require(len(utci_ok) >= 8, "thermal curve has too few populated bins")
    require(len(rain_ok) == len(RAIN_LABELS), "rainfall curve must populate every bin")
    require(len(anomaly_ok) >= 8, "anomaly curve has too few populated bins")
    peak_row = utci_ok.sort_values("mean").iloc[-1]
    hottest_row = utci_ok.sort_values("center").iloc[-1]
    anomaly_peak = anomaly_ok.sort_values("mean").iloc[-1]
    season_peak = season_ok.sort_values("mean").iloc[-1]
    season_trough = season_ok.sort_values("mean").iloc[0]

    summary = {
        "sample": {
            "all_cities": int(len(table)),
            "primary_cities": int(len(primary_table)),
            "excluded_cities": sorted(excluded_table.index.tolist()),
            "daily_rows": int(len(data)),
            "primary_rows": int(len(primary)),
            "zero_usage_days_all": int((data["usage"] == 0).sum()),
            "zero_usage_days_primary": int((primary["usage"] == 0).sum()),
            "date_min": data["date"].min().date().isoformat(),
            "date_max": data["date"].max().date().isoformat(),
        },
        "usage": {
            "median_min_city": primary_table["usage_median"].idxmin(),
            "median_min": float(primary_table["usage_median"].min()),
            "median_max_city": primary_table["usage_median"].idxmax(),
            "median_max": float(primary_table["usage_median"].max()),
            "median_spread": float(primary_table["usage_median"].max()
                                   / primary_table["usage_median"].min()),
            "log1p_min": float(primary["usage_log1p"].min()),
            "log1p_median": float(primary["usage_log1p"].median()),
            "log1p_max": float(primary["usage_log1p"].max()),
        },
        "weather": {
            "utci_city_mean_min": float(table["utci_mean_c"].min()),
            "utci_city_mean_max": float(table["utci_mean_c"].max()),
            "utci_city_mean_spread": float(table["utci_mean_c"].max()
                                           - table["utci_mean_c"].min()),
            "rain_city_mean_min": float(table["rain_mean_mm"].min()),
            "rain_city_mean_max": float(table["rain_mean_mm"].max()),
            "dry_day_share": float((data["precipitation_mm"] == 0).mean()),
            "rain_max_mm": float(data["precipitation_mm"].max()),
        },
        "thermal_response_absolute": {
            "peak_utci_c": float(peak_row["center"]),
            "peak_centered_log1p": float(peak_row["mean"]),
            "coldest_bin_centered_log1p": float(utci_ok["mean"].iloc[0]),
            "hottest_bin_utci_c": float(hottest_row["center"]),
            "hottest_bin_centered_log1p": float(hottest_row["mean"]),
            "peak_minus_hottest": float(peak_row["mean"] - hottest_row["mean"]),
            "city_band_width_at_peak": float(peak_row["q75"] - peak_row["q25"]),
            "populated_bins": int(len(utci_ok)),
            "populated_utci_range": [float(utci_ok["center"].min()),
                                     float(utci_ok["center"].max())],
        },
        "thermal_response_anomaly": {
            "peak_anomaly_c": float(anomaly_peak["center"] * 10.0),
            "peak_centered_log1p": float(anomaly_peak["mean"]),
            "coldest_bin_centered_log1p": float(anomaly_ok["mean"].iloc[0]),
            "positive_bins": int((anomaly_ok["mean"] > 0.0).sum()),
            "populated_bins": int(len(anomaly_ok)),
        },
        "rainfall_response": {
            "zero_bin_centered_log1p": float(rain_ok["mean"].iloc[0]),
            "heaviest_bin_centered_log1p": float(rain_ok["mean"].iloc[-1]),
            "heaviest_bin_vs_city_mean_change": float(np.expm1(rain_ok["mean"].iloc[-1])),
            "heaviest_vs_dry_log1p": float(rain_ok["mean"].iloc[-1] - rain_ok["mean"].iloc[0]),
            "heaviest_vs_dry_change": float(
                np.expm1(rain_ok["mean"].iloc[-1] - rain_ok["mean"].iloc[0])
            ),
            "monotone_decline": bool(np.all(np.diff(rain_ok["mean"].to_numpy()) <= 1e-9)),
        },
        "seasonality": {
            "peak_day": float(season_peak["center"]),
            "peak_centered_log1p": float(season_peak["mean"]),
            "trough_day": float(season_trough["center"]),
            "trough_centered_log1p": float(season_trough["mean"]),
            "amplitude": float(season_peak["mean"] - season_trough["mean"]),
            "seasonal_share_of_thermal_variance": {
                "min_city": min(seasonal_share, key=seasonal_share.get),
                "min": float(min(seasonal_share.values())),
                "max_city": max(seasonal_share, key=seasonal_share.get),
                "max": float(max(seasonal_share.values())),
                "mean": float(np.mean(list(seasonal_share.values()))),
                "per_city": {city: round(float(value), 6)
                             for city, value in sorted(seasonal_share.items())},
            },
        },
        "representative_cities": {
            "cities": representatives,
            "selection": "city study-period mean UTCI, equally spaced from coldest to warmest",
            "utci_mean_c": {city: float(primary.loc[primary["city"] == city,
                                                    "utci_mean_c"].mean())
                            for city in representatives},
        },
        "weekend": {
            "negative_cities": int((weekend["difference"] < 0).sum()),
            "positive_cities": int((weekend["difference"] > 0).sum()),
            "mean_difference": float(weekend["difference"].mean()),
            "min_city": weekend.index[0],
            "min_difference": float(weekend["difference"].iloc[0]),
            "max_city": weekend.index[-1],
            "max_difference": float(weekend["difference"].iloc[-1]),
        },
        "climate_groups": {
            "codes": sorted(table["climate_code"].unique().tolist()),
            "cities_per_code": {code: int((table["climate_code"] == code).sum())
                                for code in sorted(table["climate_code"].unique())},
        },
    }
    (RES_DIR / "eda_midterm.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("figures written:", len(list(FIG_DIR.glob("*.pdf"))), "pdf")


if __name__ == "__main__":
    main()
