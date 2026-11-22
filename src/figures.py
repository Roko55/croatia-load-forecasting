"""Render the four paper figures from the same frozen inputs used for scoring."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    args = parser.parse_args()
    lock = read_json(ROOT / "configs" / "protocol.json")
    sources = read_json(ROOT / "data" / "entsoe-sources.json")
    for key, fold in lock["development"].items():
        year = int(key)
        plot(year, observed(year, sources[key], args.cache),
             predictions(year, fold, lock["models"]), lock["models"])


if __name__ == "__main__":
    main()
