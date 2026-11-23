"""Build fixed electricity demand inputs for Croatian capacity-planning scenarios."""

import argparse
import csv
import gzip
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit import ROOT, digest, hours, predictions, read_json, require
from historical_sources import ZAGREB, croatian_holidays


INPUTS = ROOT / "data" / "planning"
CONFIG = ROOT / "configs" / "planning.json"
REFERENCE_YEAR = 2023
PLANNING_YEARS = (2030, 2035, 2040, 2045, 2050)
MWH_PER_PJ = 1_000_000 / 3.6
GWH_TO_MWH = 1000
ENERGY_TOLERANCE_MWH = 0.001


def scaled(weights, energy_mwh):
    """Allocate annual electricity to consecutive one-hour intervals."""
    weights = np.asarray(weights, dtype=float)
    require(math.isfinite(energy_mwh) and energy_mwh >= 0, "Invalid annual electricity")
    require(np.isfinite(weights).all() and (weights >= 0).all(), "Invalid hourly weights")
    total = math.fsum(weights)
    require(total > 0 or energy_mwh == 0, "Positive annual electricity requires a nonzero shape")
    return weights * (energy_mwh / total) if total else weights.copy()


def interpolate_anchors(anchor_rows, year):
    """Interpolate only between adjacent source anchors."""
    if str(year) in anchor_rows:
        return anchor_rows[str(year)]
    previous = max(int(key) for key in anchor_rows if int(key) < year)
    following = min(int(key) for key in anchor_rows if int(key) > year)
    fraction = (year - previous) / (following - previous)
    left, right = anchor_rows[str(previous)], anchor_rows[str(following)]
    require(left.keys() == right.keys(), "Annual anchors use different accounting categories")
    return {name: left[name] + fraction * (right[name] - left[name]) for name in left}


def annual_amounts(config, year):
    """Keep the annual energy accounting independent of the hourly forecast."""
    necp = interpolate_anchors(config["necp_pj"], year)
    tyndp = interpolate_anchors(config["tyndp_gwh"], year)
    road = necp["road"] * MWH_PER_PJ
    vehicle_total = tyndp["passenger_road"] + tyndp["commercial_road"]
    require(vehicle_total > 0, f"{year}: TYNDP vehicle split is undefined")
    passenger = road * tyndp["passenger_road"] / vehicle_total
    space_heat = (tyndp["household_space_heat"] + tyndp["service_space_heat"]) * GWH_TO_MWH
    hot_water = config["reference_gwh"]["household_hot_water"] * GWH_TO_MWH
    cooling = (tyndp["household_cooling"] + tyndp["service_cooling"]) * GWH_TO_MWH
    total = (necp["final_electricity"] + necp["network_losses"]) * MWH_PER_PJ
    base = total - road - space_heat - hot_water - cooling
    require(base > 0, f"{year}: no electricity left for the base")
    return {
        "baseline": base,
        "passenger_road": passenger,
        "commercial_road": road - passenger,
        "space_heat": space_heat,
        "hot_water": hot_water,
        "cooling": cooling,
        "total": total,
    }


def source_weights(config, clock):
    training_sources = read_json(ROOT / "data" / "training-sources.json")
    record = training_sources["weather_retrospective"]
    early_record = training_sources["weather"]
    regions = list(config["region_weights"])
    population = np.asarray(list(config["region_weights"].values()), dtype=float)
    require((population > 0).all() and math.isclose(math.fsum(population), 1.0),
            "Regional weights must be positive and sum to one")
    window = config["cooling_temperature_window_hours"]
    columns = ["timestamp_utc", *[f"temperature_{city}_c" for city in regions]]

    def weather(source):
        path = ROOT / "data" / source["file"]
        require(digest(path) == source["sha256"], f"Weather checksum differs: {path.name}")
        return pd.read_parquet(path, columns=columns).set_index("timestamp_utc")

    # The preceding hours are needed for the first cooling interval of the UTC year.
    weather_index = pd.date_range(clock[0] - pd.Timedelta(hours=window - 1), clock[-1], freq="h")
    temperature = pd.concat((weather(early_record).tail(window - 1), weather(record)))
    temperature = temperature.reindex(weather_index)
    require(np.isfinite(temperature.to_numpy()).all(), "Incomplete regional weather history")
    heat = np.maximum(config["heating_threshold_c"] - temperature.loc[clock].to_numpy(), 0)
    smoothed = temperature.rolling(window).mean().loc[clock].to_numpy()
    cool = np.maximum(smoothed - config["cooling_threshold_c"], 0)

    path = INPUTS / "local-hour-weights.csv"
    require(digest(path) == config["local_hour_weights_sha256"], "Local hourly shape checksum differs")
    shapes = pd.read_csv(path)
    require(list(shapes.columns) == ["hour", "passenger_weekday", "passenger_weekend",
                                     "commercial_road", "hot_water"], "Unexpected hourly input schema")
    require(shapes.hour.tolist() == list(range(24)), "Expected one shape value per local hour")
    require(np.isfinite(shapes.iloc[:, 1:].to_numpy()).all() and
            (shapes.iloc[:, 1:] >= 0).all().all(), "Invalid local-hour weights")
    local = clock.tz_convert(ZAGREB)
    holidays = set().union(*(croatian_holidays(year) for year in set(local.year)))
    nonworkday = (local.dayofweek >= 5) | np.isin(local.date, list(holidays))
    local_hour = local.hour.to_numpy()
    passenger = np.where(nonworkday, shapes.passenger_weekend.to_numpy()[local_hour],
                         shapes.passenger_weekday.to_numpy()[local_hour])
    return {
        "passenger_road": passenger,
        "commercial_road": shapes.commercial_road.to_numpy()[local_hour],
        "space_heat": heat @ population,
        "hot_water": shapes.hot_water.to_numpy()[local_hour],
        "cooling": cool @ population,
    }


def components(weights, amounts):
    return {
        "road_charging": scaled(weights["passenger_road"], amounts["passenger_road"]) +
        scaled(weights["commercial_road"], amounts["commercial_road"]),
        "heating_and_hot_water": scaled(weights["space_heat"], amounts["space_heat"]) +
        scaled(weights["hot_water"], amounts["hot_water"]),
        "cooling": scaled(weights["cooling"], amounts["cooling"]),
    }


def result_year(models, weights, amounts):
    """Remove historical overlap before fixing each model to the same annual base."""
    added = components(weights, amounts)
    added_total = sum(added.values())
    series, factors = dict(added), {}
    for name, residual in models.items():
        baseline = scaled(residual, amounts["baseline"])
        total = baseline + added_total
        require(abs(math.fsum(total) - amounts["total"]) < ENERGY_TOLERANCE_MWH,
                f"{name}: total annual electricity does not close")
        series[f"{name}__baseline"] = baseline
        series[f"{name}__total"] = total
        factors[name] = amounts["baseline"] / math.fsum(residual)
    return series, factors


def write_hourly(path, timestamps, series):
    """Write a portable table with reproducible gzip metadata and MW columns."""
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["reference_timestamp_utc", *[f"{name}_mw" for name in series]])
            writer.writerows((stamp, *(f"{v:.8f}" for v in values))
                             for stamp, values in zip(timestamps, zip(*series.values(), strict=True), strict=True))


def build(output):
    config = read_json(CONFIG)
    protocol = read_json(ROOT / "configs" / "protocol.json")
    reference = config["reference_gwh"]
    timestamps = hours(REFERENCE_YEAR)
    clock = pd.to_datetime(timestamps, utc=True)
    weights = source_weights(config, clock)
    archive = protocol["retrospective"][str(REFERENCE_YEAR)]
    models = predictions(REFERENCE_YEAR, archive, protocol["models"])
    historical = components(weights, {
        "passenger_road": reference["road"] * GWH_TO_MWH,
        "commercial_road": 0,
        "space_heat": (reference["household_space_heat"] + reference["service_space_heat"]) * GWH_TO_MWH,
        "hot_water": reference["household_hot_water"] * GWH_TO_MWH,
        "cooling": (reference["household_cooling"] + reference["service_cooling"]) * GWH_TO_MWH,
    })
    overlap = sum(historical.values())
    residuals = {name: np.asarray(values) - overlap for name, values in models.items()}
    for name, residual in residuals.items():
        require(np.isfinite(residual).all() and (residual >= 0).all(),
                f"{name}: historical sector deduction leaves a negative or invalid hourly base")

    output.mkdir(parents=True, exist_ok=True)
    reference_path = output / "reference-2023.csv.gz"
    write_hourly(reference_path, timestamps, {
        **{f"deducted_{name}": values for name, values in historical.items()},
        **{f"{name}__residual": values for name, values in residuals.items()},
    })
    manifest = {
        "reference_year": REFERENCE_YEAR,
        "reference_hours": len(timestamps),
        "clock": "Reference year UTC, one-hour intervals",
        "forecasts_sha256": archive["forecast_export_sha256"],
        "configuration_sha256": digest(CONFIG),
        "local_hour_weights_sha256": config["local_hour_weights_sha256"],
        "reference_sha256": digest(reference_path),
        "deducted_mwh": {name: math.fsum(values) for name, values in historical.items()},
        "minimum_reference_residual_mw": {name: float(values.min()) for name, values in residuals.items()},
        "years": {},
    }
    for year in PLANNING_YEARS:
        amounts = annual_amounts(config, year)
        series, factors = result_year(residuals, weights, amounts)
        path = output / f"{year}.csv.gz"
        write_hourly(path, timestamps, series)
        manifest["years"][year] = {"sha256": digest(path), "annual_mwh": amounts, "baseline_factors": factors}

    (output / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "planning",
                        help="Directory for scenario tables and their manifest")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
