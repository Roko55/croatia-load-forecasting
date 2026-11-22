"""Recompute the Croatian load benchmark from frozen forecasts and ENTSO-E observations."""
import argparse
import csv
from datetime import datetime, timedelta
import gzip
import hashlib
import json
import math
from pathlib import Path
from shutil import copyfileobj
from statistics import median
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "mae_mw", "rmse_mw", "mape_pct", "energy_bias_pct", "shape_mae_mw",
    "shape_peak_abs_error_mw", "shape_ramp_mae_mw", "top_5pct_overlap_pct",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hours(year):
    start = datetime(year, 1, 1)
    length = (datetime(year + 1, 1, 1) - start).days * 24
    return [(start + timedelta(hours=offset)).isoformat(timespec="minutes") + "Z"
            for offset in range(length)]


def positive(values):
    return all(math.isfinite(value) and value > 0 for value in values)


def source_file(entry, cache):
    path = cache / entry["file"]
    cache.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        partial = path.with_suffix(".partial")
        print(f"Downloading {entry['file']} from its pinned source", flush=True)
        try:
            with urllib.request.urlopen(entry["download_url"], timeout=90) as response:
                with partial.open("wb") as stream:
                    copyfileobj(response, stream)
            require(digest(partial) == entry["sha256"], f"Unexpected SHA-256 for {entry['file']}")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    require(digest(path) == entry["sha256"], f"Unexpected archive revision: {path}")
    return path


def observed(year, entry, cache):
    path = source_file(entry, cache)
    mapped = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = stream.readline()
        delimiter = "\t" if "\t" in header else ";" if ";" in header else ","
        stream.seek(0)
        for row in csv.DictReader(stream, delimiter=delimiter):
            if row["CountryCode"].strip() == "HR":
                value = datetime.strptime(row["DateUTC"].strip().replace("/", "-"), "%d-%m-%Y %H:%M")
                stamp = value.isoformat(timespec="minutes") + "Z"
                require(stamp not in mapped, f"{year}: duplicate Croatian observation at {stamp}")
                mapped[stamp] = float(row["Value"])
    expected = hours(year)
    require(set(mapped) == set(expected), f"{year}: incomplete or unexpected Croatian observations")
    result = [mapped[stamp] for stamp in expected]
    require(positive(result), f"{year}: non-positive or non-finite Croatian observation")
    return result


def predictions(year, fold, methods):
    path = ROOT / "results" / "profiles" / f"{year}.csv.gz"
    require(digest(path) == fold["forecast_export_sha256"], f"{year}: forecast archive hash differs")
    tcn_columns = [f"tcn_seed{seed}" for seed in (42, 43, 44)]
    columns = [key for key in methods if key != "tcn_median"] + tcn_columns
    with gzip.open(path, "rt", encoding="ascii", newline="") as stream:
        rows = csv.DictReader(stream)
        require(rows.fieldnames == ["timestamp_utc", *columns], f"{year}: unexpected forecast schema")
        rows = list(rows)
    require([row["timestamp_utc"] for row in rows] == hours(year), f"{year}: forecast time grid differs")
    series = {name: [float(row[name]) for row in rows] for name in columns}
    require(all(positive(values) for values in series.values()), f"{year}: invalid forecast value")
    series["tcn_median"] = [median(point) for point in zip(*(series.pop(name) for name in tcn_columns), strict=True)]
    return series


def score(actual, pred):
    count = len(actual)
    factor = math.fsum(actual) / math.fsum(pred)
    aligned = [point * factor for point in pred]
    error = [p - a for a, p in zip(actual, pred, strict=True)]
    shape_error = [p - a for a, p in zip(actual, aligned, strict=True)]
    ramp_error = [abs(after - before) for before, after in zip(shape_error, shape_error[1:])]
    top_count = math.ceil(count * 0.05)
    top_actual = set(sorted(range(count), key=actual.__getitem__)[-top_count:])
    top_prediction = set(sorted(range(count), key=aligned.__getitem__)[-top_count:])
    return {
        "mae_mw": math.fsum(map(abs, error)) / count,
        "rmse_mw": math.sqrt(math.fsum(e * e for e in error) / count),
        "mape_pct": 100 * math.fsum(abs(1 - p / a) for a, p in zip(actual, pred, strict=True)) / count,
        "energy_bias_pct": 100 * (1 / factor - 1),
        "shape_mae_mw": math.fsum(map(abs, shape_error)) / count,
        "shape_peak_abs_error_mw": abs(max(aligned) - max(actual)),
        "shape_ramp_mae_mw": math.fsum(ramp_error) / len(ramp_error),
        "top_5pct_overlap_pct": 100 * len(top_actual & top_prediction) / top_count,
    }


def run(cache):
    lock = read_json(ROOT / "configs" / "protocol.json")
    methods = lock["models"]
    with (ROOT / "results" / "by-year.csv").open(encoding="utf-8", newline="") as stream:
        expected = {(int(row["year"]), row["model"]): row for row in csv.DictReader(stream)}
    planned = {(int(year), method) for year in lock["development"] for method in methods}
    require(len(methods) == 12 and len(planned) == 48 and set(expected) == planned, "Incomplete comparison table")
    hashes = read_json(ROOT / "data" / "entsoe-sources.json")
    results = {}
    for key, fold in lock["development"].items():
        year = int(key)
        actual = observed(year, hashes[key], cache)
        require(len(actual) == fold["forecast_hours"], f"{year}: unexpected evaluation length")
        results[key] = {}
        for method, curve in predictions(year, fold, methods).items():
            computed = score(actual, curve)
            row = expected[year, method]
            require(row["recipe"] == methods[method]["recipe"], f"{year} {method}: recipe differs")
            for metric in METRICS:
                require(abs(computed[metric] - float(row[metric])) <= 1e-8,
                        f"{year} {method} {metric}: {computed[metric]} != {row[metric]}")
            results[key][method] = computed

    with (ROOT / "results" / "summary.csv").open(encoding="utf-8", newline="") as stream:
        summary = {row["model"]: row for row in csv.DictReader(stream)}
    require(set(summary) == set(methods), "Incomplete summary")
    for method in methods:
        for metric in METRICS:
            value = math.fsum(year[method][metric] for year in results.values()) / len(results)
            require(abs(value - float(summary[method][metric])) <= 1e-8,
                    f"{method} {metric}: four-year mean differs")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output", type=Path, help="Write the recomputed per-year metrics as JSON")
    args = parser.parse_args()
    try:
        scores = run(args.cache)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    if args.output:
        args.output.write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PASS: 48 hourly profiles, 384 per-year metric checks and 96 four-year mean checks.")


if __name__ == "__main__":
    main()
