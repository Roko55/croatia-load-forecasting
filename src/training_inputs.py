"""Rebuild the frozen annual development inputs from public sources."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit import ROOT, digest, observed, read_json, require, source_file
from historical_sources import ZAGREB, croatian_holidays, historical_holidays, read_bridge, read_legacy

WEATHER = tuple(f"temperature_{city}_c" for city in ("zagreb", "split", "rijeka", "osijek"))
COVARIATES = ["timestamp_utc", "is_holiday", *WEATHER]
TRAINING = [*COVARIATES, "demand_mw"]
YEARS = (2019, 2020, 2021, 2022)
START = pd.Timestamp("2006-01-01", tz="UTC")


def boundary(year):
    return pd.Timestamp(f"{year}-01-01", tz="UTC")


def time_grid(left, right):
    return pd.date_range(left, right, freq="h", inclusive="left").as_unit("ns")


def validate(train, future, year):
    require(year in YEARS, f"Unsupported development year: {year}")
    require(list(train) == TRAINING and list(future) == COVARIATES,
            "Invalid input schema. Target-year covariates must contain only time, holiday flags and temperature.")
    for frame, first, last in ((train, START, boundary(year)), (future, boundary(year), boundary(year+1))):
        require(pd.DatetimeIndex(frame.timestamp_utc).equals(time_grid(first, last)),
                "Input is outside the configured UTC window or has a gap or duplicate.")
        require(frame.is_holiday.isin([0, 1]).all(), "Holiday flags must be binary.")
        local_day = frame.timestamp_utc.dt.tz_convert(ZAGREB).dt.normalize()
        require(frame.groupby(local_day).is_holiday.nunique().eq(1).all(),
                "Holiday flags must be constant over a local day.")
        require(np.isfinite(frame[list(WEATHER)]).all().all(), "Input temperatures must be finite.")
    require(train.demand_mw.gt(0).all() and np.isfinite(train.demand_mw).all(),
            "Historical demand must be finite and positive.")


def workbook_history(legacy, bridge):
    """The legacy workbook ends at midnight in Zagreb, one hour before UTC New Year."""
    cutover = pd.Timestamp("2016-01-01", tz=ZAGREB).tz_convert("UTC")
    original = legacy[(legacy.index >= START) & (legacy.index < cutover)]
    remaining = bridge[(bridge.index >= cutover) & (bridge.index < boundary(min(YEARS)))]
    require(cutover in remaining.index, "Bridge workbook is missing the first hour after the legacy period.")
    return pd.concat([original, remaining], verify_integrity=True)


def source_history(cache, years, manifest):
    """Read historical target observations through the year before the latest requested origin."""
    files = manifest["entsoe_workbooks"]
    legacy_path = source_file(files["Monthly-hourly-load-values_2006-2015.xlsx"], cache)
    bridge_path = source_file(files["MHLV_data-2015-2019.xlsx"], cache)
    legacy, _ = read_legacy(legacy_path)
    bridge = read_bridge(bridge_path).demand_mw
    annual = read_json(ROOT / "data" / "entsoe-sources.json")
    first_modern = min(YEARS)
    stop = max(years)
    parts = [workbook_history(legacy, bridge)]
    for year in range(first_modern, stop):
        values = observed(year, annual[str(year)], cache)
        parts.append(pd.Series(values, index=time_grid(boundary(year), boundary(year+1)), name="demand_mw"))
    return pd.concat(parts, verify_integrity=True).sort_index()


def source_covariates(manifest):
    record = manifest["weather"]
    path = ROOT / "data" / record["file"]
    require(digest(path) == record["sha256"], "Frozen weather file failed SHA-256 verification.")
    weather = pd.read_parquet(path)
    require(list(weather) == ["timestamp_utc", *WEATHER], "Unexpected weather schema.")
    require(pd.DatetimeIndex(weather.timestamp_utc).equals(time_grid(START, boundary(max(YEARS)+1))),
            "Weather time grid does not match the development period.")
    holidays = set().union(*(historical_holidays(y) if y < 2015 else croatian_holidays(y)
                             for y in range(START.year, max(YEARS)+2)))
    local_days = weather.timestamp_utc.dt.tz_convert(ZAGREB).dt.date
    weather["is_holiday"] = local_days.isin(holidays).astype("int64")
    return weather[COVARIATES]


def write_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(cache, destination, years):
    pinned = read_json(ROOT / "configs" / "protocol.json")["development"]
    manifest = read_json(ROOT / "data" / "training-sources.json")
    demand = source_history(cache, years, manifest)
    covariates = source_covariates(manifest)
    for year in years:
        origin, end = boundary(year), boundary(year+1)
        historical = covariates.loc[covariates.timestamp_utc.lt(origin)].reset_index(drop=True)
        future = covariates.loc[covariates.timestamp_utc.between(origin, end, inclusive="left")].reset_index(drop=True)
        raw = demand.reindex(historical.timestamp_utc)
        raw = raw.where(raw.gt(0))
        fill_count = int(raw.isna().sum())
        require(fill_count == 21, f"{year}: expected 21 historical gaps, found {fill_count}.")
        historical["demand_mw"] = raw.ffill(limit=2).to_numpy()
        validate(historical, future, year)
        output = destination / str(year)
        for file, frame, checksum in (("training", historical, "training_sha256"),
                                      ("covariates", future, "covariate_sha256")):
            path = output / f"{file}.parquet"
            write_frame(path, frame)
            require(digest(path) == pinned[str(year)][checksum],
                    f"{year} {file}: rebuilt file differs from the frozen SHA-256. Check sources and pinned dependencies.")
        print(f"{year}: {len(historical):,} history hours, {len(future):,} forecast hours, both SHA-256 values match.",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "inputs")
    parser.add_argument("--year", type=int, choices=YEARS, nargs="+", default=YEARS)
    args = parser.parse_args()
    prepare(args.cache, args.output, args.year)


if __name__ == "__main__":
    main()
