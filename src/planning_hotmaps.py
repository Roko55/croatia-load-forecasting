"""Apply Croatian Hotmaps day-type and temperature templates to a weather year."""

import argparse
import csv
from datetime import timedelta
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

from audit import ROOT, digest, read_json, require
from historical_sources import ZAGREB, croatian_holidays

DATA = ROOT / "data" / "planning"
REGIONS = {"zagreb": "HR04", "osijek": "HR04", "rijeka": "HR03", "split": "HR03"}
TEMPERATURE_RANGES = {"heating": (-15, 17), "cooling": (19, 40)}
KEYS = ["region", "day_type", "hour", "temperature_c"]


def thermal_weights(clock, temperature, population):
    """Read full local days and return household and service weights for each end use."""
    path = DATA / "hotmaps-hr.csv.gz"
    require(digest(path) == read_json(DATA / "hotmaps-sources.json")["extract_sha256"],
            "Hotmaps extract hash differs")
    table = pd.read_csv(path)
    require(set(population) == set(REGIONS), "Hotmaps requires the four documented regional weather sites")
    weights = np.asarray(list(population.values()))
    require((weights > 0).all() and np.isclose(weights.sum(), 1, rtol=0, atol=1e-10),
            "Regional weights must be positive and sum to one")
    local = clock.tz_convert(ZAGREB)
    days = local.date
    holidays = set().union(*(croatian_holidays(y) for y in range(days[0].year, days[-1].year + 2)))
    daytypes = [
        2 if d.weekday() == 6 or d in holidays else
        1 if d.weekday() == 5 or d + timedelta(days=1) in holidays else 0
        for d in days
    ]
    local_temperature = temperature.tz_convert(ZAGREB)
    grouped = local_temperature.groupby(local_temperature.index.date)
    averages = grouped.mean().reindex(days)
    day_hours = grouped.size().reindex(days)
    expected = clock_for_days(days)
    require(np.array_equal(day_hours.to_numpy(), expected), "Hotmaps daily temperatures need complete local days")
    require(np.isfinite(averages.to_numpy()).all(), "Invalid daily-average Hotmaps weather")
    rounded_temperature = np.rint(averages.to_numpy()).astype(int)
    output = {}
    for (use, sector), profile in table.groupby(["use", "sector"], sort=False):
        values = profile.set_index(KEYS, verify_integrity=True)["weight"]
        lower, upper = TEMPERATURE_RANGES[use]
        total = np.zeros(len(clock))
        for city, weight in population.items():
            rounded = rounded_temperature[:, averages.columns.get_loc("temperature_" + city + "_c")]
            # The Croatia extract has no 18 °C bin. Use zero for the shoulder bin
            # in this sensitivity, and reject extrapolation on the active side.
            active = rounded <= upper if use == "heating" else rounded >= lower
            keys = pd.MultiIndex.from_arrays([
                np.repeat(REGIONS[city], active.sum()), np.asarray(daytypes)[active],
                local.hour.to_numpy()[active] + 1, rounded[active],
            ])
            shape = values.reindex(keys).to_numpy()
            require(np.isfinite(shape).all() and (shape >= 0).all(),
                    f"Missing or invalid Hotmaps {use}/{sector} values for {city}")
            total[active] += weight * shape
        output[use, sector] = total
    require(set(output) == {(use, sector) for use in TEMPERATURE_RANGES for sector in ("household", "service")},
            "Incomplete Hotmaps end-use extract")
    return output



def clock_for_days(days):
    starts = pd.DatetimeIndex(days).tz_localize(ZAGREB).tz_convert("UTC")
    ends = pd.DatetimeIndex([day + timedelta(days=1) for day in days]).tz_localize(ZAGREB).tz_convert("UTC")
    return (ends - starts) / pd.Timedelta(hours=1)


def extract(source, output):
    """Copy the pinned Croatian lookup rows to a small offline data file."""
    register = read_json(DATA / "hotmaps-sources.json")
    with output.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["use", "sector", *KEYS, "weight"])
            for record in register["files"]:
                path = source / record["file"]
                require(digest(path) == record["sha256"], f"Hotmaps source hash differs: {path.name}")
                with path.open(encoding="utf-8-sig", newline="") as incoming:
                    for row in csv.DictReader(incoming):
                        if row["NUTS2_code"] in REGIONS.values():
                            writer.writerow([record["use"], record["sector"], row["NUTS2_code"],
                                             row["day_type"], row["hour"], row["temperature"], row["load"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pinned Croatian Hotmaps rows.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.source_dir, args.output)
