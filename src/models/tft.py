"""Temporal Fusion Transformer using consecutive forecast windows."""
from ._neuralforecast import direct


def forecast(train, future):
    from neuralforecast.models import TFT

    options = dict(hidden_size=16, n_head=2, dropout=.1)
    _, forecast, _ = direct("tft", TFT, options, 220, 336, train, future, 42)
    return forecast
