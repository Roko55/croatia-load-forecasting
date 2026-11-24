"""Plot published electricity demand inputs and the spread across models."""

import argparse
from pathlib import Path

import matplotlib.dates as dates
import matplotlib.pyplot as plt
import pandas as pd

from audit import ROOT
from historical_sources import ZAGREB
from planning import PLANNING_YEARS


def plot_profiles(table, output):
    profiles = pd.read_csv(table, parse_dates=["reference_timestamp_utc"])
    profiles = profiles.set_index("reference_timestamp_utc").tz_convert(ZAGREB)
    totals = [column for column in profiles if column.endswith("__total_mw")]
    baseline = "calendar_recent__baseline_mw"
    sectors = ["road_charging_mw", "heating_and_hot_water_mw", "cooling_mw"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharey=True)
    for axis, first, last, season in zip(
        axes, ("2023-01-16", "2023-07-17"), ("2023-01-22", "2023-07-23"), ("Winter", "Summer"), strict=True
    ):
        week = profiles.loc[first:last] / 1000
        axis.stackplot(
            week.index, *(week[column] for column in [baseline, *sectors]),
            labels=["Calendar base", "Road charging", "Heating and hot water", "Cooling"],
            colors=["#286777", "#E2B550", "#C67251", "#8F9BB9"], alpha=0.95,
        )
        axis.fill_between(week.index, week[totals].min(axis=1), week[totals].max(axis=1),
                          color="#304253", alpha=0.19, label="Range of 12 total profiles")
        axis.plot(week.index, week["calendar_recent__total_mw"], color="#223143", linewidth=1)
        axis.set_title(season, loc="left", fontweight="bold")
        axis.set_ylabel("Demand (GW)")
        axis.set_xlim(week.index[0], week.index[-1])
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_locator(dates.DayLocator(tz=ZAGREB))
        axis.xaxis.set_major_formatter(dates.DateFormatter("%a %d %b", tz=ZAGREB))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=5, frameon=False,
               bbox_to_anchor=(0.5, 0.017))
    fig.suptitle("Croatian electricity demand in the 2050 planning scenario",
                 x=0.08, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.08, 0.905, "Illustrative weeks on the reference 2023 clock · Croatia local time",
             color="#39495A", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.86, hspace=0.4)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def plot_planning_years(directory, output):
    """Compare seasonal demand and annual peaks across the five planning years."""
    fig, (weekly, peaks) = plt.subplots(1, 2, figsize=(12, 4.5),
                                       gridspec_kw={"width_ratios": [2, 1]})
    colours = ("#9EAEBE", "#708CA3", "#457F88", "#23656F", "#163A59")
    for year, colour in zip(PLANNING_YEARS, colours, strict=True):
        table = pd.read_csv(directory / f"{year}.csv.gz", parse_dates=["reference_timestamp_utc"])
        table = table.set_index("reference_timestamp_utc") / 1000
        weeks = table["calendar_recent__total_mw"].resample("7D")
        calendar = weeks.mean().where(weeks.count() == 168).dropna()
        weekly.plot(calendar.index, calendar, label=str(year), color=colour, linewidth=2)
        model_peaks = table.filter(regex="__total_mw$").max().sort_values()
        peaks.vlines(year, model_peaks.iloc[0], model_peaks.iloc[-1], color=colour, alpha=.65)
        peaks.scatter([year] * len(model_peaks), model_peaks, color=colour, s=19)

    weekly.set_title("Weekly demand | Calendar model", loc="left")
    weekly.set_ylabel("Mean demand (GW)")
    weekly.set_xlim(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC"))
    weekly.xaxis.set_major_locator(dates.MonthLocator(interval=2))
    weekly.xaxis.set_major_formatter(dates.DateFormatter("%b"))
    weekly.legend(ncols=5, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.26))
    peaks.set_title("Annual peaks | All 12 models", loc="left")
    peaks.set_ylabel("Peak demand (GW)")
    peaks.set_xticks(PLANNING_YEARS)
    for axis in (weekly, peaks):
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Croatian electricity demand for H2RES", x=.08, ha="left",
                 fontsize=17, fontweight="bold")
    fig.text(.08, .88, "Fixed charging and common 2023 weather", fontsize=10, color="#39495A")
    fig.subplots_adjust(left=.08, right=.98, bottom=.19, top=.77, wspace=.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=ROOT / "results" / "planning" / "2050.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "planning-2050.png")
    parser.add_argument("--annual-output", type=Path,
                        help="Plot all planning years using the directory that contains --table")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_profiles(args.table, args.output)
    if args.annual_output is not None:
        plot_planning_years(args.table.parent, args.annual_output)


if __name__ == "__main__":
    main()
