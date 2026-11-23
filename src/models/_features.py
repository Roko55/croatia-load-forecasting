"""Shared calendar features and trailing temperature averages."""
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training_inputs import WEATHER


def calendar(frame, weather):
    """Encode Zagreb time and use the holiday flag already supplied for each hour."""
    local = pd.DatetimeIndex(frame.timestamp_utc).tz_convert(ZoneInfo("Europe/Zagreb"))
    if not frame.is_holiday.isin([0, 1]).all():
        raise ValueError("is_holiday must be binary, all local hours marked by the calendar input")
    x = frame[list(weather) + ["is_holiday"]].astype(float).copy()
    for key, values, period in (("hour", local.hour, 24), ("week", local.dayofweek, 7),
                                ("annual", (local.dayofyear - 1) / np.where(local.is_leap_year, 366, 365), 1)):
        x[key + "_sin"] = np.sin(2 * np.pi * values / period)
        x[key + "_cos"] = np.cos(2 * np.pi * values / period)
    x["weekend"] = (local.dayofweek >= 5).astype(float)
    return x


def trailing_temperatures(train, future, windows=(24,)):
    """Append trailing temperature averages across the forecast boundary."""
    history = train[list(WEATHER)]
    combined = pd.concat([history, future[list(WEATHER)]], ignore_index=True)
    feature_names = list(WEATHER)
    for window in windows:
        means = combined.rolling(window, min_periods=1).mean().add_suffix(f"_avg{window}")
        train = train.assign(**dict(zip(means.columns, means.iloc[:len(train)].to_numpy().T, strict=True)))
        future = future.assign(**dict(zip(means.columns, means.iloc[len(train):].to_numpy().T, strict=True)))
        np.testing.assert_allclose(
            means.iloc[len(train)].to_numpy(),
            combined.iloc[max(0, len(train) - window + 1):len(train) + 1].mean().to_numpy(),
            rtol=0, atol=1e-10,
        )
        feature_names.extend(means.columns)
    assert "demand_mw" not in future
    return train, future, tuple(feature_names)
