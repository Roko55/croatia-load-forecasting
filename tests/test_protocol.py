import unittest

import pandas as pd

from audit import ROOT, read_json, score
from training_inputs import (
    COVARIATES, START, WEATHER, boundary, source_covariates, time_grid, validate,
    workbook_history,
)


class HistoricalLoadTest(unittest.TestCase):
    def test_workbook_cutover_preserves_last_utc_hour_of_2015(self):
        index = pd.date_range("2015-12-31 22:00", periods=3, freq="h", tz="UTC").as_unit("ns")
        legacy = pd.Series([2015.0, 9000.0], index=index[:2])
        bridge = pd.Series([8000.0, 1886.0, 2016.0], index=index)
        result = workbook_history(legacy, bridge)
        pd.testing.assert_series_equal(result, pd.Series([2015.0, 1886.0, 2016.0], index=index))

    def test_missing_bridge_boundary_is_rejected(self):
        legacy = pd.Series([1.0], index=pd.DatetimeIndex(["2015-12-31 22:00Z"]))
        bridge = pd.Series([2.0], index=pd.DatetimeIndex(["2016-01-01 00:00Z"]))
        with self.assertRaisesRegex(ValueError, "first hour after the legacy period"):
            workbook_history(legacy, bridge)


class ForecastInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = cls.frame(START, boundary(2019)).assign(demand_mw=2000.0)
        cls.future = cls.frame(boundary(2019), boundary(2020))

    @staticmethod
    def frame(left, right):
        return pd.DataFrame({"timestamp_utc": time_grid(left, right), "is_holiday": 0,
                             **dict.fromkeys(WEATHER, 10.0)})[COVARIATES]

    def test_correct_grid_passes(self):
        validate(self.train, self.future, 2019)

    def test_future_load_columns_are_rejected(self):
        for field in ("demand_mw", "demand_lag_24"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Invalid input schema"):
                validate(self.train, self.future.assign(**{field: 1000.0}), 2019)

    def test_load_at_origin_is_rejected(self):
        late = self.train.copy()
        late.loc[late.index[-1], "timestamp_utc"] = boundary(2019)
        with self.assertRaisesRegex(ValueError, "configured UTC window"):
            validate(late, self.future, 2019)

    def test_missing_or_duplicate_utc_hour_is_rejected(self):
        missing = self.future.iloc[1:]
        duplicate = pd.concat([self.future.iloc[:1], self.future.iloc[:-1]], ignore_index=True)
        for name, frame in (("missing", missing), ("duplicate", duplicate)):
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, "configured UTC window"):
                validate(self.train, frame, 2019)

    def test_frozen_weather_keeps_dst_and_leap_hours(self):
        source = read_json(ROOT / "data" / "training-sources.json")
        frame = source_covariates(source)
        local = frame.timestamp_utc.dt.tz_convert("Europe/Zagreb")
        self.assertEqual(int(local.dt.strftime("%Y-%m-%d").eq("2020-03-29").sum()), 23)
        self.assertEqual(int(local.dt.strftime("%Y-%m-%d").eq("2020-10-25").sum()), 25)
        self.assertEqual(int(frame.timestamp_utc.dt.strftime("%Y-%m-%d").eq("2020-02-29").sum()), 24)
        christmas = frame.loc[local.dt.strftime("%Y-%m-%d").eq("2020-12-25"), "is_holiday"]
        self.assertEqual(len(christmas), 24)
        self.assertTrue(christmas.eq(1).all())


class ScoringTest(unittest.TestCase):
    def test_annual_rescaling_removes_a_uniform_level_error(self):
        result = score([100.0, 200.0], [110.0, 220.0])
        self.assertAlmostEqual(result["mae_mw"], 15.0)
        self.assertAlmostEqual(result["energy_bias_pct"], 10.0)
        for key in ("shape_mae_mw", "shape_peak_abs_error_mw", "shape_ramp_mae_mw"):
            self.assertAlmostEqual(result[key], 0.0)

    def test_peak_timing_is_distinct_from_peak_magnitude(self):
        result = score([100.0, 200.0], [200.0, 100.0])
        self.assertAlmostEqual(result["energy_bias_pct"], 0.0)
        self.assertAlmostEqual(result["shape_peak_abs_error_mw"], 0.0)
        self.assertAlmostEqual(result["top_5pct_overlap_pct"], 0.0)
        self.assertAlmostEqual(result["shape_mae_mw"], 100.0)
        self.assertAlmostEqual(result["shape_ramp_mae_mw"], 200.0)
