"""Chronos-2 with a pinned checkpoint and historical level calibration."""
import pandas as pd

from audit import digest
from training_inputs import WEATHER

from ._features import calendar

CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def forecast(train, future, checkpoint):
    import torch
    from chronos import Chronos2Pipeline

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config = {'checkpoint': 'amazon/chronos-2', 'revision': CHRONOS_REVISION,
              'dtype': 'float32', 'context_hours': 8192, 'batch_size': 16, 'point': 'median',
              'unrolled_quantiles': [0.5], 'level_history_years': 3, 'level_weight': 0.5}
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
    values = pred['0.5'].to_numpy(dtype=float)
    origin = future.timestamp_utc.iloc[0]
    past_mean = train.loc[train.timestamp_utc >= origin - pd.DateOffset(years=3), "demand_mw"].mean()
    values *= 0.5 + 0.5 * past_mean / values.mean()
    return values, {'configuration': config, 'effective_context_rows': len(context),
                                              'checkpoint_sha256': weight_sha}
