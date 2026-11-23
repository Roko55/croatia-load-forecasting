"""Temporal convolutional network. The output is the hourly median of three seeds."""
import numpy as np
import pandas as pd

from training_inputs import WEATHER

from ._features import calendar


def forecast(torch, train, future):
    """Use the hourly median to limit the influence of an outlying seed."""
    trials = [tcn_fit(torch, train, future, seed) for seed in (42, 43, 44)]
    return np.median(np.stack(trials), axis=0)


def previous_year_features(train, future):
    """For each UTC year, derive demand features only from its prior UTC year.

    Use prior-year mean load and the deviation of each prior-year calendar bin
    from that mean. The first year has no previous year and receives a missing
    indicator rather than an estimate computed from its own load.
    """
    if "demand_mw" in future:
        raise ValueError("target-year demand must not be supplied for generation")
    if train.timestamp_utc.iloc[-1] >= future.timestamp_utc.iloc[0]:
        raise ValueError("origin must follow the end of observed training demand")
    if future.timestamp_utc.dt.year.nunique() != 1:
        raise ValueError("expected a single forecast year")
    stamps = pd.concat([train.timestamp_utc, future.timestamp_utc], ignore_index=True)
    years = stamps.dt.year.to_numpy()
    local = stamps.dt.tz_convert("Europe/Zagreb")
    hours = local.dt.hour.to_numpy()
    index = pd.MultiIndex.from_arrays([local.dt.month, local.dt.dayofweek, hours])
    historic_load = train.demand_mw.to_numpy(dtype=float)
    historical_years = years[:len(train)]
    features = np.zeros((len(stamps), 3))
    for year in np.unique(years):
        previous = historical_years == year - 1
        if not previous.any():
            continue
        scheduled = years == year
        annual_mean = historic_load[previous].mean()
        load = pd.Series(historic_load[previous], index=index[:len(train)][previous])
        by_calendar = load.groupby(level=[0, 1, 2]).mean()
        by_hour = load.groupby(level=2).mean()
        expected = by_calendar.reindex(index[scheduled]).to_numpy()
        fallback = by_hour.reindex(hours[scheduled]).fillna(annual_mean).to_numpy()
        features[scheduled, 0] = annual_mean
        features[scheduled, 1] = np.where(np.isnan(expected), fallback, expected) - annual_mean
        features[scheduled, 2] = 1
    return features[:len(train)], features[len(train):]


def tcn_fit(torch, train, future, seed):
    """Fit three causal convolutions on masked chunks and predict a full year."""
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    class CausalConv(nn.Module):
        def __init__(self, inputs, dilation):
            super().__init__()
            self.left = 2 * dilation
            self.conv = nn.Conv1d(inputs, 8, 3, dilation=dilation)

        def forward(self, x):
            return F.gelu(self.conv(F.pad(x, (self.left, 0))))

    past_load, future_load = previous_year_features(train, future)
    x = np.column_stack([calendar(train, WEATHER).to_numpy(), past_load])
    f = np.column_stack([calendar(future, WEATHER).to_numpy(), future_load])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    x, f = ((x - mean) / scale).astype('float32'), ((f - mean) / scale).astype('float32')
    y = train.demand_mw.to_numpy()
    y_mean, y_scale = y.mean(), y.std()
    y = ((y - y_mean) / y_scale).astype('float32')
    context, step = 24, 672
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(28):
        for indices in torch.randperm(count).split(8):
            optimizer.zero_grad()
            error = (model(xt[indices])[:, 0, context:] - yt[indices]).abs() * mt[indices]
            loss = error.sum() / mt[indices].sum()
            if not torch.isfinite(loss):
                raise ValueError('TCN loss diverged')
            loss.backward()
            optimizer.step()
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
    return pred
