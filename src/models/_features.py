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


def trailing_temperatures(train, future):
    """Use the current and previous 23 temperatures, including across Jan 1."""
    columns = tuple(f'{name}_avg24' for name in WEATHER)
    history = train[list(WEATHER)]
    combined = pd.concat([history, future[list(WEATHER)]], ignore_index=True)
    means = combined.rolling(24, min_periods=1).mean()
    train = train.assign(**dict(zip(columns, means.iloc[:len(train)].to_numpy().T)))
    future = future.assign(**dict(zip(columns, means.iloc[len(train):].to_numpy().T)))
    np.testing.assert_allclose(
        future.loc[0, list(columns)].to_numpy(dtype=float),
        np.vstack([history.to_numpy()[-23:], future.loc[0, list(WEATHER)].to_numpy(dtype=float)]).mean(axis=0),
        rtol=0, atol=1e-10,
    )
    assert 'demand_mw' not in future
    return train, future, (*WEATHER, *columns)
