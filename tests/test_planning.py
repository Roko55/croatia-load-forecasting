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
