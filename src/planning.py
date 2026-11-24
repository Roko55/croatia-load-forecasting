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


def annual_amounts(config, year, hot_water_case="continuity"):
    """Keep the annual energy accounting independent of the hourly forecast."""
    necp = interpolate_anchors(config["necp_pj"], year)
    tyndp = interpolate_anchors(config["tyndp_gwh"], year)
    road = necp["road"] * MWH_PER_PJ
    vehicle_total = tyndp["passenger_road"] + tyndp["commercial_road"]
    require(vehicle_total > 0, f"{year}: TYNDP vehicle split is undefined")
    passenger = road * tyndp["passenger_road"] / vehicle_total
    space_heat = (tyndp["household_space_heat"] + tyndp["service_space_heat"]) * GWH_TO_MWH
    hot_water = (
        tyndp["household_hot_water_sensitivity"] if hot_water_case == "tyndp"
        else config["reference_gwh"]["household_hot_water"]
    ) * GWH_TO_MWH
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


def regional_weather(config):
    """Load the pinned temperatures, including both boundaries of the 2023 UTC slice."""
    training_sources = read_json(ROOT / "data" / "training-sources.json")
    columns = ["timestamp_utc", *[f"temperature_{city}_c" for city in config["region_weights"]]]
    parts = []
    for key in ("weather", "weather_retrospective"):
        record = training_sources[key]
        path = ROOT / "data" / record["file"]
        require(digest(path) == record["sha256"], f"Weather checksum differs: {path.name}")
        parts.append(pd.read_parquet(path, columns=columns).set_index("timestamp_utc"))
    return pd.concat(parts, verify_integrity=True)


def source_weights(config, clock, temperature=None):
    regions = list(config["region_weights"])
    population = np.asarray(list(config["region_weights"].values()), dtype=float)
    require((population > 0).all() and math.isclose(math.fsum(population), 1.0),
            "Regional weights must be positive and sum to one")
    window = config["cooling_temperature_window_hours"]
    # The preceding hours are needed for the first cooling interval of the UTC year.
    weather_index = pd.date_range(clock[0] - pd.Timedelta(hours=window - 1), clock[-1], freq="h")
    if temperature is None:
        temperature = regional_weather(config)
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


def seasonal_passenger_weights(clock, source):
    """Apply TYNDP seasonal day shapes on the common Croatian calendar."""
    local = clock.tz_convert(ZAGREB)
    holidays = set().union(*(croatian_holidays(y) for y in set(local.year)))
    result = []
    for stamp in local:
        season = "summer" if stamp.month in source["warm_months"] else "winter"
        day_type = "weekend" if stamp.weekday() >= 5 else "weekday"
        if stamp.date() in holidays:
            day_type = "holiday"
        result.append(source["profiles"][f"{season}_{day_type}"][stamp.hour])
    return np.asarray(result)


def components(weights, amounts):
    return {
        "road_charging": scaled(weights["passenger_road"], amounts["passenger_road"]) +
        scaled(weights["commercial_road"], amounts["commercial_road"]),
        "heating_and_hot_water": scaled(weights["space_heat"], amounts["space_heat"]) +
        scaled(weights["hot_water"], amounts["hot_water"]),
        "cooling": scaled(weights["cooling"], amounts["cooling"]),
    }


def result_year(models, added, amounts, flexible_mwh=0):
    """Remove historical overlap before fixing each model to the same annual base."""
    added_total = sum(added.values())
    series, factors = dict(added), {}
    for name, residual in models.items():
        baseline = scaled(residual, amounts["baseline"])
        total = baseline + added_total
        require(abs(math.fsum(total) + flexible_mwh - amounts["total"]) < ENERGY_TOLERANCE_MWH,
                f"{name}: fixed and flexible annual electricity does not close")
        series[f"{name}__baseline"] = baseline
        field = "fixed_demand" if flexible_mwh else "total"
        series[f"{name}__{field}"] = total
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


def apply_hotmaps(added, profile, hotmaps, sector_gwh, weights, hot_water_mwh):
    """Set alternative thermal shapes while preserving separate sector energies."""
    added["cooling"] = sum(
        scaled(hotmaps["cooling", sector], sector_gwh[f"{sector}_cooling"] * GWH_TO_MWH)
        for sector in ("household", "service")
    )
    if profile == "hotmaps-thermal":
        added["heating_and_hot_water"] = scaled(weights["hot_water"], hot_water_mwh) + sum(
            scaled(hotmaps["heating", sector], sector_gwh[f"{sector}_space_heat"] * GWH_TO_MWH)
            for sector in ("household", "service")
        )


def build(output, thermal_profile="degree-hours", rebase_thermal=False, hot_water_case="continuity",
          ev_case="fixed", passenger_shape="reference"):
    require(thermal_profile in ("degree-hours", "hotmaps-cooling", "hotmaps-thermal"), "Unknown thermal profile")
    require(hot_water_case in ("continuity", "tyndp"), "Unknown hot-water energy case")
    require(ev_case in ("fixed", "tyndp-flex"), "Unknown passenger charging case")
    require(passenger_shape in ("reference", "tyndp-seasonal"), "Unknown passenger charging shape")
    require(not rebase_thermal or thermal_profile != "degree-hours", "Rebasing requires a Hotmaps profile")
    config = read_json(CONFIG)
    protocol = read_json(ROOT / "configs" / "protocol.json")
    reference = config["reference_gwh"]
    timestamps = hours(REFERENCE_YEAR)
    clock = pd.to_datetime(timestamps, utc=True)
    temperature = regional_weather(config)
    weights = source_weights(config, clock, temperature)
    if thermal_profile != "degree-hours":
        from planning_hotmaps import thermal_weights

        hotmaps = thermal_weights(clock, temperature, config["region_weights"])
    if ev_case == "tyndp-flex":
        from planning_ev import PUBLISHED

        ev = read_json(PUBLISHED)
        shares = {str(r["YEAR"]): {"flex_share": r["EV_flex_share"] / 100}
                  for r in ev["EV_FLEX_SHARE"]}
    archive = protocol["retrospective"][str(REFERENCE_YEAR)]
    models = predictions(REFERENCE_YEAR, archive, protocol["models"])
    historical = components(weights, {
        "passenger_road": reference["road"] * GWH_TO_MWH,
        "commercial_road": 0,
        "space_heat": (reference["household_space_heat"] + reference["service_space_heat"]) * GWH_TO_MWH,
        "hot_water": reference["household_hot_water"] * GWH_TO_MWH,
        "cooling": (reference["household_cooling"] + reference["service_cooling"]) * GWH_TO_MWH,
    })
    future_weights = weights
    if passenger_shape == "tyndp-seasonal":
        passenger_file = INPUTS / "tyndp-hr-passenger-shapes.json"
        future_weights = dict(weights, passenger_road=seasonal_passenger_weights(clock, read_json(passenger_file)))
    if rebase_thermal:
        apply_hotmaps(historical, thermal_profile, hotmaps, reference, weights,
                      reference["household_hot_water"] * GWH_TO_MWH)
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
    if (thermal_profile, hot_water_case, ev_case, passenger_shape) != (
        "degree-hours", "continuity", "fixed", "reference"
    ):
        manifest["sensitivity"] = {
            "thermal_profile": thermal_profile,
            "rebase_thermal": rebase_thermal,
            "hot_water_case": hot_water_case,
            "ev_case": ev_case,
            "passenger_shape": passenger_shape,
        }
        if thermal_profile != "degree-hours":
            manifest["sensitivity"]["hotmaps_sha256"] = digest(INPUTS / "hotmaps-hr.csv.gz")
        if ev_case == "tyndp-flex":
            manifest["sensitivity"]["tyndp_ev_sha256"] = digest(PUBLISHED)
        if passenger_shape == "tyndp-seasonal":
            manifest["sensitivity"]["passenger_shapes_sha256"] = digest(passenger_file)
    for year in PLANNING_YEARS:
        amounts = annual_amounts(config, year, hot_water_case)
        flexible_mwh = 0
        if ev_case == "tyndp-flex":
            share = interpolate_anchors(shares, year)["flex_share"]
            require(0 < share < 1, f"{year}: invalid TYNDP flexible share")
            flexible_mwh = amounts["passenger_road"] * share
        added = components(future_weights, dict(amounts, passenger_road=amounts["passenger_road"] - flexible_mwh))
        if thermal_profile != "degree-hours":
            sector_gwh = interpolate_anchors(config["tyndp_gwh"], year)
            apply_hotmaps(added, thermal_profile, hotmaps, sector_gwh, weights, amounts["hot_water"])
        series, factors = result_year(residuals, added, amounts, flexible_mwh)
        path = output / f"{year}.csv.gz"
        write_hourly(path, timestamps, series)
        manifest["years"][year] = {"sha256": digest(path), "annual_mwh": amounts, "baseline_factors": factors}
        if ev_case == "tyndp-flex":
            manifest["years"][year]["flexible_passenger_grid_mwh"] = flexible_mwh
            manifest["years"][year]["fixed_demand_mwh"] = amounts["total"] - flexible_mwh

    (output / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "planning",
                        help="Directory for scenario tables and their manifest")
    parser.add_argument("--thermal-profile", choices=("degree-hours", "hotmaps-cooling", "hotmaps-thermal"),
                        default="degree-hours")
    parser.add_argument("--rebase-thermal", action="store_true",
                        help="Use the selected Hotmaps profiles in the 2023 deduction as well")
    parser.add_argument("--hot-water", choices=("continuity", "tyndp"), default="continuity",
                        help="Annual electricity trajectory for household hot water")
    parser.add_argument("--ev-case", choices=("fixed", "tyndp-flex"), default="fixed",
                        help="Export fixed load with a separate grid energy budget for flexible charging")
    parser.add_argument("--passenger-shape", choices=("reference", "tyndp-seasonal"), default="reference",
                        help="Use TYNDP's six seasonal and day-type shapes for future passenger charging")
    args = parser.parse_args()
    build(args.output, args.thermal_profile, args.rebase_thermal, args.hot_water, args.ev_case, args.passenger_shape)


if __name__ == "__main__":
    main()
