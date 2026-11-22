"""Temporal convolutional network. The output is the hourly median of three seeds."""
import numpy as np

from training_inputs import WEATHER

from ._features import calendar


def forecast(torch, train, future):
    trials = [tcn_control(torch, train, future, seed)[0] for seed in (42, 43, 44)]
    return np.median(np.stack(trials), axis=0)


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
