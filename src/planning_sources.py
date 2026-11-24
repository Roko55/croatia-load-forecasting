"""Extract the local-hour templates from the cited TYNDP, NLR and When2Heat files."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, time, timedelta
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean
import zipfile

from openpyxl import load_workbook

from audit import ROOT, digest, read_json, require
from historical_sources import ZAGREB

INPUTS = ROOT / "data" / "planning"


def passenger_series(path, hashes):
    """Sum fixed market and prosumer charging on the source clock."""
    by_variant = []
    with zipfile.ZipFile(path) as archive:
        for variant in ("MARKET", "PROSUMER"):
            name = f"EV_FIXED_LOAD_PROFILES_ELECTRICITY_{variant} 2030.xlsx"
            raw = archive.read(f"Demand/2030/{name}")
            require(hashlib.sha256(raw).hexdigest() == hashes[variant], f"TYNDP workbook differs: {name}")
            workbook = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
            rows = workbook["HR00"].values
            for _ in range(10):
                next(rows)
            column = next(rows).index("WS003")
            values = [(row[0], row[1], row[column]) for row in rows]
            workbook.close()
            require(len(values) == 8760 and all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0
                                               for _, _, v in values), f"Invalid TYNDP profile: {name}")
            start = datetime(2001, 1, 1)
            for index, (day, hour, _) in enumerate(values):
                expected_day = start + timedelta(days=index // 24)
                require(day == expected_day.strftime("%d.%m.") and hour == index % 24 + 1,
                        f"Unexpected interval order in {name}")
            by_variant.append(values)
    series = []
    for market, prosumer in zip(*by_variant, strict=True):
        require(market[:2] == prosumer[:2], "TYNDP variants use different interval labels")
        series.append(market[2] + prosumer[2])
    return series


def passenger_shapes(series):
    """Build the two typical days used in the published reference case."""
    weekdays = [[] for _ in range(24)]
    weekends = [[] for _ in range(24)]
    for index, value in enumerate(series):
        day, hour = divmod(index, 24)
        if 7 <= day <= 13:
            (weekdays if day <= 11 else weekends)[hour].append(value)
    return [mean(group) for group in weekdays], [mean(group) for group in weekends]


def seasonal_passenger_shapes(series):
    """Extract the six distinct days in the 2030 HR00 fixed charging source."""
    source_dates = {
        "winter_weekday": (1, 8), "winter_weekend": (1, 13), "winter_holiday": (1, 1),
        "summer_weekday": (5, 2), "summer_weekend": (5, 5), "summer_holiday": (5, 1),
    }
    days = [series[i:i + 24] for i in range(0, len(series), 24)]
    start = datetime(2018, 1, 1)
    templates = {key: days[(datetime(2018, *date) - start).days] for key, date in source_dates.items()}
    require(len({tuple(x) for x in templates.values()}) == 6 and
            all(day in templates.values() for day in days), "Unexpected daily TYNDP charging shapes")
    return {"source_dates": source_dates, "warm_months": [5, 6, 7, 8, 9, 10], "profiles": templates}


def seconds(value):
    stamp = time.fromisoformat(value)
    return stamp.hour * 3600 + stamp.minute * 60 + stamp.second


def commercial_shape(path):
    """Allocate each vehicle-day's mileage over its reported off-shift intervals."""
    days = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            days[row["veh_op_day_id"]].append(row)
    shape = [0.0] * 24
    for rows in days.values():
        mileage = math.fsum(float(row["vmt"]) for row in rows)
        parking = [(seconds(row["start_time"]), seconds(row["end_time"]))
                   for row in rows if row["on_shift"] == "0"]
        duration = sum(end - start for start, end in parking)
        require(mileage > 0 and duration > 0, "A depot vehicle-day has no mileage or charging opportunity")
        for hour in range(24):
            overlap = sum(max(0, min(end, (hour + 1) * 3600) - max(start, hour * 3600))
                          for start, end in parking)
            shape[hour] += mileage * overlap / duration
    return shape


def hot_water_shape(path, house_share, expected_sha256=None):
    """Combine normalized Croatian profiles by local hour of day in 2022."""
    groups = [[[] for _ in range(24)] for _ in range(2)]
    start = datetime.fromisoformat("2021-12-31T23:00:00+00:00").timestamp()
    # The published CSV omits the second local 02:00 on the autumn DST transition.
    omitted_hour = datetime.fromisoformat("2022-10-30T01:00:00+00:00").timestamp()
    expected = iter(start + i * 3600 for i in range(8760) if start + i * 3600 != omitted_hour)
    count = 0
    selected = hashlib.sha256()
    columns = ("HR_heat_profile_water_SFH", "HR_heat_profile_water_MFH")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            if not "2021-12-31T23:00:00Z" <= row["utc_timestamp"] < "2022-12-31T23:00:00Z":
                continue
            timestamp = datetime.fromisoformat(row["utc_timestamp"])
            require(timestamp.timestamp() == next(expected, None), "Unexpected When2Heat 2022 interval")
            fields = (row["utc_timestamp"], *(row[column] for column in columns))
            selected.update((";".join(fields) + "\n").encode("ascii"))
            values = [float(field.replace(",", ".")) for field in fields[1:]]
            require(all(math.isfinite(v) and v >= 0 for v in values), "Invalid When2Heat hourly weight")
            for group, value in zip(groups, values, strict=True):
                group[timestamp.astimezone(ZAGREB).hour].append(value)
            count += 1
    require(count == 8759, "Unexpected When2Heat 2022 hour count")
    checksum = selected.hexdigest()
    if expected_sha256 is not None:
        require(checksum == expected_sha256, "When2Heat 2022 source values differ")
    means = [[mean(values) for values in group] for group in groups]
    single, multi = ([v / math.fsum(group) for v in group] for group in means)
    require(0 <= house_share <= 1, "Invalid population share in houses")
    return [house_share * a + (1 - house_share) * b for a, b in zip(single, multi, strict=True)], checksum


def write_weights(path, weekday, weekend, commercial, hot_water):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["hour", "passenger_weekday", "passenger_weekend", "commercial_road", "hot_water"])
        writer.writerows((hour, *(f"{v:.12f}" for v in row))
                         for hour, row in enumerate(zip(weekday, weekend, commercial, hot_water, strict=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tyndp-demand", type=Path, required=True)
    parser.add_argument("--nlr-schedules", type=Path, required=True)
    parser.add_argument("--when2heat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seasonal-passenger-output", type=Path)
    args = parser.parse_args()
    sources = read_json(INPUTS / "sources.json")
    require(digest(args.nlr_schedules) == sources["commercial_road"]["sha256"],
            "NLR schedules do not match the cited source")
    passenger = passenger_series(args.tyndp_demand, sources["passenger_road"]["workbook_sha256"])
    weekday, weekend = passenger_shapes(passenger)
    hot_water, _ = hot_water_shape(args.when2heat, sources["hot_water"]["house_population_share"],
                                   sources["hot_water"]["selected_rows_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_weights(args.output, weekday, weekend, commercial_shape(args.nlr_schedules), hot_water)
    if args.seasonal_passenger_output:
        args.seasonal_passenger_output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(seasonal_passenger_shapes(passenger), indent=2) + "\n"
        args.seasonal_passenger_output.write_bytes(payload.encode("utf-8"))


if __name__ == "__main__":
    main()
