"""Neural workflows with historical targets and known future covariates."""
from importlib.metadata import version

import numpy as np
import pandas as pd

from classical import calendar
from training_inputs import WEATHER

STEPS = {"nhits": 260, "tft": 220}


def inputs(data):
    features = calendar(data, WEATHER)
    features["ds"] = data.timestamp_utc.dt.tz_localize(None)
    features["unique_id"] = "HR"
    return features


def block_starts(count, horizon):
    starts = list(range(0, count - horizon + 1, horizon))
    if starts[-1] + horizon < count:
        starts.append(count - horizon)
    return starts


def direct(model, horizon, train, future, seed):
    import logging
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import NHITS, TFT

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
    cls, options = {
        "nhits": (NHITS, dict(n_blocks=[1, 1, 1], mlp_units=[[64, 64]] * 3,
                            n_pool_kernel_size=[12, 6, 1], n_freq_downsample=[24, 6, 1])),
        "tft": (TFT, dict(hidden_size=16, n_head=2, dropout=.1)),
    }[model]
    neural = cls(h=horizon, input_size=horizon, futr_exog_list=features, exclude_insample_y=True,
                 scaler_type="identity", random_seed=seed, step_size=24, max_steps=STEPS[model],
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
    starts = block_starts(len(known), horizon)
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
                  "context_hours": horizon, "steps": STEPS[model], "features": features,
                  "seed": seed, "target_center_mw": center, "target_scale_mw": spread,
                  "feature_means": mean.to_dict(), "feature_scales": scale.to_dict(),
                  "runtime": {name: version(name) for name in ("neuralforecast", "torch", "numpy", "pandas")},
                  "target_feedback_invariance": "exact", "excludes_insample_y": True}


def centered_days(engine, train, future, receipt):
    horizon, day = 168, 24
    history, known = inputs(train), inputs(future)
    features = receipt["features"]
    center, scale = pd.Series(receipt["feature_means"]), pd.Series(receipt["feature_scales"])
    for frame in (history, known):
        frame[features] = (frame[features] - center) / scale
    source = pd.concat([history.tail(horizon), known], ignore_index=True)
    days = range(0, len(known), day)
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


def tcn_control(torch, train, future, seed):
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    config = {'seed': seed, 'channels': 8, 'dilations': [1, 3, 8], 'kernel': 3,
              'epochs': 14, 'batch_chunks': 8, 'chunk_hours': 672, 'loss': 'L1',
              'learning_rate': .002, 'optimizer': 'AdamW', 'weather_only_history_hours': 24}

    class CausalConv(nn.Module):
        def __init__(self, inputs, dilation):
            super().__init__()
            self.left = 2 * dilation
            self.conv = nn.Conv1d(inputs, 8, 3, dilation=dilation)

        def forward(self, x):
            return F.gelu(self.conv(F.pad(x, (self.left, 0))))

    x = calendar(train, WEATHER).to_numpy()
    f = calendar(future, WEATHER).to_numpy()
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    x, f = ((x - mean) / scale).astype('float32'), ((f - mean) / scale).astype('float32')
    y = train.demand_mw.to_numpy()
    y_mean, y_scale = y.mean(), y.std()
    y = ((y - y_mean) / y_scale).astype('float32')
    context, step = 24, config['chunk_hours']
    padded = np.concatenate([np.repeat(x[:1], context, axis=0), x])
    # Fixed-size packs with masks retain the last partial training chunk.
    count = (len(x) + step - 1) // step
    inputs = np.zeros((count, x.shape[1], step + context), dtype='float32')
    labels = np.zeros((count, step), dtype='float32')
    mask = np.zeros_like(labels)
    for i, offset in enumerate(range(0, len(x), step)):
        available = min(step, len(x) - offset)
        inputs[i, :, :available + context] = padded[offset:offset + available + context].T
        labels[i, :available], mask[i, :available] = y[offset:offset + available], 1
    assert mask.sum() == len(train)
    xt, yt, mt = map(torch.from_numpy, (inputs, labels, mask))
    model = nn.Sequential(CausalConv(x.shape[1], 1), CausalConv(8, 3),
                          CausalConv(8, 8), nn.Conv1d(8, 1, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    history = []
    for _ in range(config['epochs']):
        total = 0.
        for indices in torch.randperm(count).split(config['batch_chunks']):
            optimizer.zero_grad()
            error = (model(xt[indices])[:, 0, context:] - yt[indices]).abs() * mt[indices]
            loss = error.sum() / mt[indices].sum()
            if not torch.isfinite(loss):
                raise ValueError('TCN loss diverged')
            loss.backward()
            optimizer.step()
            total += error.detach().sum().item()
        history.append(total / len(train))
    model.eval()
    with torch.inference_mode():
        cov = torch.from_numpy(np.concatenate([x[-context:], f]).T.copy())[None]
        original = model(cov)[0, 0, context:]
        cut = len(f) // 2
        perturbed = cov.clone()
        perturbed[:, :, context + cut:] += 1
        countercheck = model(perturbed)[0, 0, context:]
        torch.testing.assert_close(original[:cut], countercheck[:cut], rtol=0, atol=0)
        pred = original.numpy().astype(float) * y_scale + y_mean
    return pred, {'configuration': config, 'train_loss': history, 'torch': torch.__version__,
                  'causality_check': 'unchanged before perturbation of later covariates'}


def lstm(torch, train, future, seed):
    from torch import nn

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    center = calendar(train, WEATHER).to_numpy()
    target = calendar(future, WEATHER).to_numpy()
    mean, sd = center.mean(0), center.std(0).clip(min=1e-8)
    x, f = ((v - mean).__truediv__(sd).astype("float32") for v in (center, target))
    y = train.demand_mw.to_numpy()
    y_mean, y_sd = y.mean(), y.std()
    y = ((y - y_mean) / y_sd).astype("float32")
    context, step = 24, 336

    def pack(data, prefix):
        chunks = (len(data)+step-1)//step
        padded = np.r_[prefix, data, np.zeros((chunks*step-len(data), data.shape[1]), dtype=data.dtype)]
        return np.stack([padded[i*step:i*step+context+step] for i in range(chunks)])

    a = torch.from_numpy(pack(x, np.repeat(x[:1], context, axis=0)))
    b = torch.from_numpy(np.pad(y, (0, len(a)*step-len(y))).reshape(-1, step))
    mask = torch.arange(len(a)*step).reshape(-1, step) < len(y)
    net = nn.LSTM(a.shape[-1], 16, batch_first=True)
    head = nn.Linear(16, 1)
    opt = torch.optim.AdamW([*net.parameters(), *head.parameters()], lr=.002)

    def model(t):
        return head(net(t)[0])[:, context:, 0]

    for _ in range(12):
        for indices in torch.randperm(len(a)).split(16):
            opt.zero_grad()
            diff = (model(a[indices]) - b[indices]).abs()[mask[indices]]
            loss = diff.mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite LSTM training loss")
            loss.backward()
            opt.step()
    net.eval()
    head.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(pack(f, x[-context:]))).flatten()[:len(f)].numpy()
    return prediction.astype(float)*y_sd+y_mean, {"seed":seed, "epochs":12, "hidden_units":16, "chunk_hours":step,
                                                   "warmup_hours":context, "uses_future_load":False}
