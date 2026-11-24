import csv
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit import ROOT, digest, hours
from planning import PLANNING_YEARS, build, scaled


def read_table(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


class PlanningTests(unittest.TestCase):
    def test_energy_allocation(self):
        np.testing.assert_allclose(scaled([1, 2, 0], 90), [30, 60, 0])
        np.testing.assert_array_equal(scaled([0, 0], 0), [0, 0])
        for values in ([0, 0], [1, -1], [1, float("nan")]):
            with self.assertRaises(ValueError):
                scaled(values, 90)

    def test_published_profiles_match_generator_and_balance(self):
        archive = ROOT / "results" / "planning"
        manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
        protocol = json.loads((ROOT / "configs" / "protocol.json").read_text(encoding="utf-8"))
        timestamps = hours(2023)
        self.assertEqual(tuple(map(int, manifest["years"])), PLANNING_YEARS)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            build(destination)
            for year in PLANNING_YEARS:
                file = f"{year}.csv.gz"
                receipt = manifest["years"][str(year)]
                self.assertEqual(digest(archive / file), receipt["sha256"])
                expected, actual = read_table(archive / file), read_table(destination / file)
                self.assertEqual([row["reference_timestamp_utc"] for row in actual], timestamps)
                self.assertEqual(actual[0].keys(), expected[0].keys())
                columns = {key: np.array([float(row[key]) for row in actual])
                           for key in actual[0] if key.endswith("_mw")}
                for key, values in columns.items():
                    np.testing.assert_allclose(values, [float(row[key]) for row in expected],
                                               rtol=0, atol=0.00000002)
                    self.assertGreaterEqual(values.min(), 0)
                road = columns["road_charging_mw"]
                heat = columns["heating_and_hot_water_mw"]
                cool = columns["cooling_mw"]
                amounts = receipt["annual_mwh"]
                for group, keys in ((road, ("passenger_road", "commercial_road")),
                                    (heat, ("space_heat", "hot_water")), (cool, ("cooling",))):
                    self.assertAlmostEqual(math.fsum(group), sum(amounts[key] for key in keys), places=3)
                for model in protocol["models"]:
                    base = columns[f"{model}__baseline_mw"]
                    total = columns[f"{model}__total_mw"]
                    np.testing.assert_allclose(total, base + road + heat + cool, rtol=0, atol=0.00000003)
                    self.assertAlmostEqual(math.fsum(base), amounts["baseline"], places=3)
                    self.assertAlmostEqual(math.fsum(total), amounts["total"], places=3)
            reference = "reference-2023.csv.gz"
            self.assertEqual(digest(archive / reference), manifest["reference_sha256"])
            self.assertEqual(read_table(archive / reference), read_table(destination / reference))
            self.assertGreater(min(manifest["minimum_reference_residual_mw"].values()), 0)

    def test_hotmaps_cooling_changes_only_the_future_cooling_addition(self):
        archive = ROOT / "results" / "planning"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            build(destination, thermal_profile="hotmaps-cooling")
            reference = "reference-2023.csv.gz"
            self.assertEqual(read_table(archive / reference), read_table(destination / reference))
            previous = read_table(archive / "2050.csv.gz")
            updated = read_table(destination / "2050.csv.gz")
            for field in ("road_charging_mw", "heating_and_hot_water_mw", "calendar_recent__baseline_mw"):
                self.assertEqual([row[field] for row in previous], [row[field] for row in updated])
            original_cooling = [float(row["cooling_mw"]) for row in previous]
            updated_cooling = [float(row["cooling_mw"]) for row in updated]
            self.assertAlmostEqual(math.fsum(original_cooling), math.fsum(updated_cooling), places=3)
            self.assertAlmostEqual(max(updated_cooling), 1236.13, places=1)

    def test_hotmaps_rebase_and_flexible_passenger_energy(self):
        archive = ROOT / "results" / "planning"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            build(destination, thermal_profile="hotmaps-thermal", rebase_thermal=True,
                  hot_water_case="tyndp", ev_case="tyndp-flex", passenger_shape="tyndp-seasonal")
            reference = "reference-2023.csv.gz"
            self.assertNotEqual(read_table(archive / reference), read_table(destination / reference))
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertGreater(min(manifest["minimum_reference_residual_mw"].values()), 0)
            for year, receipt in manifest["years"].items():
                amounts = receipt["annual_mwh"]
                self.assertEqual(receipt["flexible_passenger_grid_mwh"], amounts["passenger_road"] / 2)
                rows = read_table(destination / f"{year}.csv.gz")
                for model in receipt["baseline_factors"]:
                    self.assertNotIn(f"{model}__total_mw", rows[0])
                    fixed = math.fsum(float(row[f"{model}__fixed_demand_mw"]) for row in rows)
                    self.assertAlmostEqual(fixed, receipt["fixed_demand_mwh"], places=3)
            self.assertAlmostEqual(manifest["years"]["2050"]["annual_mwh"]["hot_water"], 466307.91, places=2)

    def test_hotmaps_all_replaces_hot_water_shape_at_equal_annual_energy(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            build(destination, thermal_profile="hotmaps-all", passenger_shape="tyndp-seasonal")
            reference = ROOT / "results" / "planning"
            self.assertEqual(read_table(reference / "reference-2023.csv.gz"),
                             read_table(destination / "reference-2023.csv.gz"))
            rows = read_table(destination / "2050.csv.gz")
            heat = [float(row["heating_and_hot_water_mw"]) for row in rows]
            road = [float(row["road_charging_mw"]) for row in rows]
            receipt = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))["years"]["2050"]
            self.assertAlmostEqual(math.fsum(heat), 3832680.015, places=3)
            self.assertAlmostEqual(max(heat), 1664.02, places=1)
            self.assertAlmostEqual(math.fsum(road), 4850000, places=3)
            self.assertAlmostEqual(max(road), 1567.28, places=1)
            self.assertEqual(receipt["annual_mwh"]["hot_water"], 1236390)
