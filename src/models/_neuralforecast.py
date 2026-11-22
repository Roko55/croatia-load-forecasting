"""Shared fitting and fixed-window inference for N-HiTS and TFT."""
from importlib.metadata import version

import numpy as np
import pandas as pd

from training_inputs import WEATHER

from ._features import calendar


def inputs(data):
    features = calendar(data, WEATHER)
    features["ds"] = data.timestamp_utc.dt.tz_localize(None)
    features["unique_id"] = "HR"
    return features


def block_starts(count, horizon):
    if horizon < 1 or count < horizon:
        raise ValueError(f"Forecast requires at least one complete {horizon}-hour window (got {count} hours).")
    starts = list(range(0, count - horizon + 1, horizon))
    if starts[-1] + horizon < count:
        starts.append(count - horizon)
    return starts


def direct(model, cls, options, steps, horizon, train, future, seed):
    starts = block_starts(len(future), horizon)

    import logging
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    history, known = inputs(train), inputs(future)
    features = [name for name in known if name not in ("unique_id", "ds")]
    mean, scale = history[features].mean(), history[features].std(ddof=0).replace(0, 1)
    for frame in (history, known):
        frame[features] = (frame[features] - mean) / scale
        assert np.isfinite(frame[features].to_numpy()).all()
    target = train.demand_mw.to_numpy(dtype=np.float64)
    center, spread = target.mean(), target.std()
    history["y"] = ((target - center) / spread).astype(np.float32)
    neural = cls(h=horizon, input_size=horizon, futr_exog_list=features, exclude_insample_y=True,
                 scaler_type="identity", random_seed=seed, step_size=24, max_steps=steps,
                 windows_batch_size=8, valid_batch_size=4, inference_windows_batch_size=1,
                 learning_rate=.001, loss=MAE(), accelerator="cpu", devices=1, logger=False,
                 enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False,
                 alias=model, **options)
    assert neural.exclude_insample_y
    engine = NeuralForecast(models=[neural], freq="h")
    engine.fit(df=history, val_size=0, verbose=False)
    assert engine.models[0].exclude_insample_y

    # Every forecast block is a separate series with covariates and a constant dummy target.
    source = pd.concat([history[known.columns].tail(horizon), known], ignore_index=True)
    contexts, blocks = [], []
    for index in starts:
        tag = f"block{index:05d}"
        context = source.iloc[index:index + horizon].assign(unique_id=tag, y=np.float32(0))
        block = known.iloc[index:index + horizon].assign(unique_id=tag)
        assert context.ds.iloc[-1] + pd.Timedelta(hours=1) == block.ds.iloc[0]
        contexts.append(context)
        blocks.append(block)
    combined = pd.concat(blocks, ignore_index=True)
    profiles = engine.predict(df=pd.concat(contexts, ignore_index=True), futr_df=combined, verbose=False)
    assert len(profiles) == len(combined)
    assert not profiles.duplicated(["unique_id", "ds"]).any()

    # The fitted model must produce identical predictions even if a context's dummy y is perturbed.
    probe = blocks[0]
    altered = contexts[0].assign(y=np.linspace(-10, 10, horizon, dtype=np.float32))
    reference = engine.predict(df=contexts[0], futr_df=probe, verbose=False)
    probe_result = engine.predict(df=altered, futr_df=probe, verbose=False)
    np.testing.assert_array_equal(probe_result[model], reference[model])

    pred = np.empty(len(known), dtype=float)
    seen = 0
    for index, block in zip(starts, blocks, strict=True):
        profile = profiles.loc[profiles.unique_id == block.unique_id.iloc[0]]
        np.testing.assert_array_equal(profile.ds, block.ds)
        pred[seen:index + horizon] = profile[model].to_numpy(dtype=float)[seen - index:] * spread + center
        seen = index + horizon
    assert seen == len(known)
    np.testing.assert_allclose(profiles.loc[profiles.unique_id == probe.unique_id.iloc[0], model],
                               reference[model], rtol=2e-5, atol=2e-6)
    return engine, pred, {"class": cls.__name__, "options": options, "horizon_hours": horizon,
                  "context_hours": horizon, "steps": steps, "features": features,
                  "seed": seed, "target_center_mw": center, "target_scale_mw": spread,
                  "feature_means": mean.to_dict(), "feature_scales": scale.to_dict(),
                  "runtime": {name: version(name) for name in ("neuralforecast", "torch", "numpy", "pandas")},
                  "target_feedback_invariance": "exact", "excludes_insample_y": True}
