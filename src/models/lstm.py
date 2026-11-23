"""Recurrent LSTM with a 24-hour weather and calendar context."""
import numpy as np

from training_inputs import WEATHER

from ._features import calendar


def forecast(torch, train, future, seed):
    """Fit on 14-day blocks with 24 hours of warm-up covariates per block."""
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
        # The warm-up initializes the recurrent state. Return the following 14 days.
        return head(net(t)[0])[:, context:, 0]

    for _ in range(18):
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
        # Join the forecast blocks in time order and trim padding after the last forecast hour.
        prediction = model(torch.from_numpy(pack(f, x[-context:]))).flatten()[:len(f)].numpy()
    return prediction.astype(float)*y_sd+y_mean, {"seed":seed, "epochs":18, "hidden_units":16, "chunk_hours":step,
                                                   "warmup_hours":context, "uses_future_load":False}
