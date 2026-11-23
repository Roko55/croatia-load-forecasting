import unittest

import numpy as np
import pandas as pd

from models.tcn import previous_year_features


class PriorYearLoadTest(unittest.TestCase):
    def test_each_year_uses_only_the_preceding_year(self):
        times = pd.to_datetime(
            ["2019-01-01", "2019-01-02", "2020-01-01", "2020-01-02"], utc=True
        )
        train = pd.DataFrame({"timestamp_utc": times, "demand_mw": [1000, 1100, 2000, 2200]})
        future = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2021-01-01"], utc=True)})
        history, forecast = previous_year_features(train, future)

        np.testing.assert_array_equal(history[:2], np.zeros((2, 3)))
        np.testing.assert_array_equal(history[2:, 0], [1050, 1050])
        np.testing.assert_array_equal(forecast[:, 0], [2100])

        changed = train.assign(demand_mw=[1000, 1100, 3000, 3300])
        changed_history, changed_forecast = previous_year_features(changed, future)
        np.testing.assert_array_equal(changed_history, history)
        np.testing.assert_array_equal(changed_forecast[:, 0], [3150])

    def test_historical_feature_builder_rejects_future_demand(self):
        train = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2020-12-31"], utc=True),
            "demand_mw": [1000],
        })
        future = train.assign(timestamp_utc=pd.to_datetime(["2021-01-01"], utc=True))
        with self.assertRaisesRegex(ValueError, "target-year demand"):
            previous_year_features(train, future)
