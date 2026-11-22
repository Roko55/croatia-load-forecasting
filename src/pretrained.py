"""Forecast with pinned Chronos-2 and TimesFM-3 checkpoints."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd

from audit import digest, read_json, ROOT
from classical import calendar
from neural import inputs
from training_inputs import WEATHER

CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def checkpoint_spec():
    return read_json(ROOT / "configs" / "checkpoints.json")


def chronos(train, future, checkpoint):
    import torch
    from chronos import Chronos2Pipeline

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config = {'checkpoint': 'amazon/chronos-2', 'revision': CHRONOS_REVISION,
              'dtype': 'float32', 'context_hours': 8192, 'batch_size': 16, 'point': 'median',
              'unrolled_quantiles': [0.5]}
    weight_sha = digest(checkpoint / 'model.safetensors')
    pipe = Chronos2Pipeline.from_pretrained(
        checkpoint, local_files_only=True,
        torch_dtype=torch.float32, trust_remote_code=False)
    assert next(pipe.model.parameters()).device.type == 'cpu'

    def frame(source):
        result = calendar(source, WEATHER)
        result['timestamp'] = source.timestamp_utc.dt.tz_localize(None)
        result['id'] = 'HR'
        return result

    context = frame(train.iloc[-8192:].reset_index(drop=True))
    context['target'] = train.demand_mw.to_numpy()[-8192:]
    target = frame(future)
    with torch.inference_mode():
        pred = pipe.predict_df(context, future_df=target, target='target',
                               id_column='id', timestamp_column='timestamp',
                               prediction_length=len(target), quantile_levels=[.5],
                               context_length=8192, batch_size=16,
                               unrolled_quantiles=config['unrolled_quantiles'])
    assert pd.DatetimeIndex(pred.timestamp).as_unit('ns').equals(
        pd.DatetimeIndex(target.timestamp).as_unit('ns'))
    return pred['0.5'].to_numpy(dtype=float), {'configuration': config, 'effective_context_rows': len(context),
                                              'checkpoint_sha256': weight_sha}


@contextmanager
def bounded_attention(torch):
    from timesfm3.torch.transformer import MultiHeadAttention

    full = MultiHeadAttention.forward
    proof = {}

    def chunks(self, tensor, options, width):
        outputs, masks = [], []
        for index in range(0, len(tensor), width):
            section = slice(index, index + width)
            local = {key: value[section] if value is not None else None for key, value in options.items()}
            output, cache, mask = full(self, tensor[section], **local)
            assert cache is None
            outputs.append(output)
            masks.append(mask)
        return torch.cat(outputs), None, torch.cat(masks)

    def forward(self, inputs_q, **options):
        width = 2 if inputs_q.shape[1] > 256 else 96
        if len(inputs_q) <= width:
            return full(self, inputs_q, **options)
        assert options.pop("decode_cache", None) is None
        if not proof:
            fixture = inputs_q[:4, :64]
            local = {key: value[:4, :64] if value is not None else None for key, value in options.items()}
            baseline, _, mask = full(self, fixture, **local)
            partitioned, _, other_mask = chunks(self, fixture, local, 2)
            torch.testing.assert_close(partitioned, baseline, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(other_mask, mask, rtol=0, atol=0)
            proof.update({"fixture_max_abs": (partitioned - baseline).abs().max().item(),
                          "same_attention_axis": True})
        return chunks(self, inputs_q, options, width)

    with patch.object(MultiHeadAttention, "forward", forward):
        yield proof


def timesfm_context(torch, train, future, folder):
    from safetensors.torch import load_file
    from timesfm3.torch.model import TimesFM3Torch
    from timesfm3.torch.transformer import RotaryPositionalEmbedding
    from torch.nn.attention import SDPBackend, sdpa_kernel

    receipt = checkpoint_spec()["timesfm3"]
    path = folder / "model.safetensors"
    cfg = read_json(folder / "config.json")
    cover = 16384
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
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), bounded_attention(torch) as proof:
        pred = model.decode(target, horizon=len(future), past_future_covariates=covs)
    assert proof.get("same_attention_axis")
    assert pred.shape == (1, 1 + len(features), len(future), len(cfg["quantiles"]))
    values = pred[0, 0, :, cfg["quantiles"].index(.5)].numpy()
    return values, {
        "past_context_hours": cover, "checkpoint_sha256": receipt["sha256"], "revision": receipt["revision"],
        "known_future_features": features, "point_quantile": .5, "attention_batch_check": proof}
