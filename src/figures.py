"""Render weekly profiles and the hourly example from the frozen scoring inputs."""
import argparse
from datetime import datetime, timedelta
from math import ceil, floor, fsum
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from audit import ROOT, observed, predictions, read_json

OBSERVED = "#27323b"
FORECAST = "#087a8a"
WEEK = 7 * 24


def weekly(values):
    chunks = [values[start:start + WEEK] for start in range(0, len(values), WEEK)]
    return [fsum(chunk) / len(chunk) for chunk in chunks]


def plot(year, actual, curves, models):
    target = weekly(actual)
    forecasts = {model: weekly(values) for model, values in curves.items()}
    scale = target + [point for series in forecasts.values() for point in series]
    limits = (250 * floor(min(scale) / 250), 250 * ceil(max(scale) / 250))
    origin = datetime(year, 1, 1)
    dates = [origin + timedelta(hours=start + min(WEEK, len(actual) - start) / 2)
             for start in range(0, len(actual), WEEK)]

    figure, axes = plt.subplots(4, 3, figsize=(12, 8.5), sharex=True, sharey=True)
    figure.subplots_adjust(left=.086, right=.989, bottom=.09, top=.826, hspace=.39, wspace=.12)
    figure.text(.063, .96, f"Croatian electricity demand profiles in {year}", fontsize=19,
                fontweight="semibold", color=OBSERVED)
    figure.text(.064, .925, "Weekly mean power from hourly observations and frozen forecasts",
                fontsize=11, color="#475569")
    figure.legend(handles=[Line2D([], [], color=OBSERVED, label="Observed ENTSO-E"),
                           Line2D([], [], color=FORECAST, linestyle="--", label="Forecast")],
                  loc="upper left", bbox_to_anchor=(.055, .902), ncol=2, frameon=False,
                  fontsize=10)
    figure.supylabel("Weekly mean demand (MW)", x=.019, fontsize=10)
    figure.text(.063, .035, "Seven-day UTC blocks from January 1. The last block may be shorter.",
                fontsize=9, color="#475569")
    for axis, (model, meta) in zip(axes.flat, models.items(), strict=True):
        label = meta["label"] + (" (pretrained)" if meta["type"] == "external pretrained" else "")
        axis.set_title(label, loc="left", fontsize=10, color=OBSERVED, pad=5)
        axis.plot(dates, target, color=OBSERVED, linewidth=1.25)
        axis.plot(dates, forecasts[model], color=FORECAST, linewidth=1.25, linestyle="--")
        axis.set_ylim(*limits)
        axis.set_xlim(origin, datetime(year + 1, 1, 1))
        axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10), bymonthday=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axis.grid(axis="y", linewidth=.55, color="#d8e1e6")
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#c4ccd2")
        axis.tick_params(labelsize=8, length=0, pad=4, colors="#475569")
    figure.savefig(ROOT / "results" / "figures" / f"profiles-{year}.png", dpi=150, facecolor="white")
    plt.close(figure)


def plot_peak_week(year, actual, curves):
    peak = max(range(len(actual)), key=actual.__getitem__)
    start = max(0, min(peak - WEEK // 2, len(actual) - WEEK))
    end = start + WEEK
    origin = datetime(year, 1, 1)
    dates = [origin + timedelta(hours=hour) for hour in range(start, end)]
    peak_time = origin + timedelta(hours=peak)
    selection = (
        ("ffn", "MLP", "#087a8a", "-"),
        ("lightgbm_thermal", "LightGBM", "#b65324", "--"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.095, right=.97, bottom=.115, top=.78, hspace=.42)
    fig.text(.095, .955, f"Croatian electricity demand around the {year} peak",
             fontsize=17, fontweight="semibold", color=OBSERVED)
    fig.text(.095, .91, f"Hourly load over seven days around the observed peak on {peak_time:%d %b, %H:%M} UTC",
             fontsize=10, color="#475569")

    for ax, scaled in zip(axes, (False, True), strict=True):
        ax.set_title("Equal annual energy" if scaled else "Raw forecasts",
                     loc="left", fontsize=11, fontweight="semibold", color=OBSERVED)
        ax.axvline(peak_time, color="#7b8894", linestyle=":", linewidth=1)
        ax.plot(dates, actual[start:end], color=OBSERVED, linewidth=1.6, label="Observed ENTSO-E")
        for key, label, color, style in selection:
            forecast = curves[key]
            factor = fsum(actual) / fsum(forecast) if scaled else 1
            ax.plot(dates, [load * factor for load in forecast[start:end]],
                    label=label, color=color, linestyle=style, linewidth=1.3)
        ax.set_ylabel("Load (MW)")
        ax.grid(axis="y", linewidth=.55, color="#d8e1e6")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].set_xlabel("Date (UTC)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.089, .885), ncol=3, frameon=False)
    fig.savefig(ROOT / "results" / "figures" / f"peak-week-{year}.png", dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    args = parser.parse_args()
    lock = read_json(ROOT / "configs" / "protocol.json")
    sources = read_json(ROOT / "data" / "entsoe-sources.json")
    for key, fold in lock["development"].items():
        year = int(key)
        actual = observed(year, sources[key], args.cache)
        curves = predictions(year, fold, lock["models"])
        plot(year, actual, curves, lock["models"])
        if year == 2022:
            plot_peak_week(year, actual, curves)


if __name__ == "__main__":
    main()
