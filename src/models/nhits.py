"""N-HiTS, retaining the central 24 hours of overlapping forecast windows."""
import numpy as np
import pandas as pd

from ._neuralforecast import direct, inputs


def forecast(train, future):
    from neuralforecast.models import NHITS

    options = dict(n_blocks=[1, 1, 1], mlp_units=[[64, 64]] * 3,
                   n_pool_kernel_size=[12, 6, 1], n_freq_downsample=[24, 6, 1])
    engine, _, settings = direct("nhits", NHITS, options, 260, 168, train, future, 42)
    return centered_days(engine, train, future, settings)


def centered_days(engine, train, future, receipt):
    """Use the centre day of each weekly forecast where the year boundaries allow it."""
    horizon, day = 168, 24
    history, known = inputs(train), inputs(future)
    features = receipt["features"]
    center, scale = pd.Series(receipt["feature_means"]), pd.Series(receipt["feature_scales"])
    for frame in (history, known):
        frame[features] = (frame[features] - center) / scale
    source = pd.concat([history.tail(horizon), known], ignore_index=True)
    days = range(0, len(known), day)
    # Keep each day away from window edges where possible, and keep every
    # weekly window within the target year.
    starts = [min(max(i - (horizon-day)//2, 0), len(known)-horizon) for i in days]
    contexts, blocks = [], []
    for start in sorted(set(starts)):
        tag = f"block{start:05d}"
        contexts.append(source.iloc[start:start+horizon].assign(unique_id=tag, y=np.float32(0)))
        blocks.append(known.iloc[start:start+horizon].assign(unique_id=tag))
    forecast = engine.predict(df=pd.concat(contexts, ignore_index=True),
                              futr_df=pd.concat(blocks, ignore_index=True), verbose=False)
    assert not forecast.duplicated(["unique_id", "ds"]).any()
    by_id = {tag: window.set_index("ds").nhits for tag, window in forecast.groupby("unique_id", sort=False)}
    values = np.concatenate([by_id[f"block{start:05d}"].loc[known.ds.iloc[i:i+day]].to_numpy(dtype=float)
                             for i, start in zip(days, starts, strict=True)])
    return values * receipt["target_scale_mw"] + receipt["target_center_mw"]
