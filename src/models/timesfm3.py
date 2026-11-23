"""TimesFM-3 inference with a pinned checkpoint and 12,288 history hours."""
import numpy as np
import pandas as pd

from audit import ROOT, read_json
from training_inputs import WEATHER

from ._neuralforecast import inputs


def forecast(torch, train, future, folder):
    from safetensors.torch import load_file
    from timesfm3.torch.model import TimesFM3Torch
    from timesfm3.torch.transformer import RotaryPositionalEmbedding
    from torch.nn.attention import SDPBackend, sdpa_kernel

    receipt = read_json(ROOT / "configs" / "checkpoints.json")["timesfm3"]
    path = folder / "model.safetensors"
    cfg = read_json(folder / "config.json")
    cover = 12288
    past, known = (inputs(frame) for frame in (train.tail(cover).reset_index(drop=True), future))
    features = [*WEATHER, "hour_sin", "hour_cos", "weekend", "is_holiday"]
    matrix = pd.concat([past[features], known[features]], ignore_index=True).to_numpy(dtype=np.float32).T
    covs = torch.tensor(np.ascontiguousarray(matrix))[None]
    target = torch.tensor(train.demand_mw.to_numpy(dtype=np.float32)[-cover:])[None, None]
    with torch.device("meta"):
        model = TimesFM3Torch(**cfg)
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True, assign=True)
    for module in model.modules():
        if isinstance(module, RotaryPositionalEmbedding) and module.timescale.is_meta:
            module.timescale = RotaryPositionalEmbedding(module.embedding_dims, module.min_timescale,
                                                         module.max_timescale).timescale
    assert all(not tensor.is_meta for tensor in model.buffers())
    model.eval()
    # Fix the kernel across runs while retaining the same attention mask and weights.
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        pred = model.decode(target, horizon=len(future), past_future_covariates=covs)
    assert pred.shape == (1, 1 + len(features), len(future), len(cfg["quantiles"]))
    values = pred[0, 0, :, cfg["quantiles"].index(.5)].numpy()
    return values, {
        "past_context_hours": cover, "checkpoint_sha256": receipt["sha256"], "revision": receipt["revision"],
        "known_future_features": features, "point_quantile": .5, "attention_backend": "math"}
