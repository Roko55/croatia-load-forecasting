"""Extract the Croatian passenger-EV inputs published with TYNDP 2026."""

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

from audit import ROOT, digest, require

SOURCE_URL = "https://2026-data.entsos-tyndp-scenarios.eu/input/model-input/national-trends/Electric-Vehicle.zip"
SOURCE_SHA256 = "4b9afb883d9a1bd7a98faccc80381424191c73f4c6e73325718fac838638fcca"
PUBLISHED = ROOT / "data" / "planning" / "tyndp-hr-ev.json"
YEARS = (2030, 2035, 2040, 2050)


def extract(archive_path):
    from openpyxl import load_workbook

    require(digest(archive_path) == SOURCE_SHA256, "TYNDP Electric Vehicle archive hash differs")
    with ZipFile(archive_path) as archive:
        def table(name):
            file = f"Electric-Vehicle/{name}.xlsx"
            with io.BytesIO(archive.read(file)) as data:
                book = load_workbook(data, read_only=True, data_only=True)
                rows = list(book.active.values)
                book.close()
            width = sum(field is not None for field in rows[0])
            return [dict(zip(rows[0][:width], row[:width], strict=True)) for row in rows[1:]]

        flex = [row for row in table("EV_FLEX_SHARE") if row["Node"] == "HR00" and row["SCENARIO"] == "All"]
        require(tuple(row["YEAR"] for row in flex) == YEARS, "Unexpected Croatian TYNDP flexibility years")
        profiles = {}
        for name in ("DRIVING_PROFILES", "AVAILABILITY_PROFILES", "EV_MIN_SOC"):
            values = table(name)
            require([r["Hour"] for r in values] == list(range(1, 25)), f"Unexpected hours in {name}")
            require(all(0 <= v <= 1 for r in values for key, v in r.items() if key != "Hour"),
                    f"Invalid fraction in {name}")
            profiles[name] = values
        return {
            "source": SOURCE_URL,
            "source_sha256": SOURCE_SHA256,
            "attribution": "TYNDP 2026 Scenarios",
            "license": "CC BY 4.0",
            "market_zone": "HR00",
            "EV_FLEX_SHARE": flex,
            "EV_ETM_DATA": [r for r in table("EV_ETM_DATA") if r["COUNTRY"] == "HR"
                            and r["EV TYPE"] == "Passenger" and r["SCENARIO"] == "NT"],
            "EV_CHARGING STATIONS": [r for r in table("EV_CHARGING STATIONS") if r["NODE"] == "All"],
            "EV_MISC_PARAMETERS": table("EV_MISC_PARAMETERS"),
            **profiles,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = extract(args.archive)
    args.output.write_bytes((json.dumps(data, indent=2) + "\n").encode("utf-8"))
