"""Historical calendar baseline with a four-year half-life."""
import numpy as np
import pandas as pd

from ._statistical import calendar


def forecast(train, future, half_life=4):
    def keys(frame):
        hour, dow, month, _ = calendar(frame)
        return pd.DataFrame({"month": month, "type": np.where(frame.is_holiday, 7, dow), "hour": hour})

    obs, pred = keys(train), keys(future)
    w = np.exp2(-np.arange(len(train)-1, -1, -1)/(half_life*365.25*24))
    obs["weight"], obs["weighted_load"] = w, w*train.demand_mw
    result = np.full(len(pred), np.average(train.demand_mw, weights=w))
    for group in (["hour"], ["type", "hour"], ["month", "type", "hour"]):
        totals = obs.groupby(group)[["weight", "weighted_load"]].sum()
        profiles = (totals.weighted_load / totals.weight).rename("value")
        joined = pred.join(profiles, on=group).value.to_numpy()
        result = np.where(np.isfinite(joined), joined, result)
    return result
