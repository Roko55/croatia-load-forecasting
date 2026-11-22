"""Penalized generalized additive model with weekday and weekend hour effects."""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from ._features import trailing_temperatures
from ._statistical import Config, PenalizedGAM, calendar, periodic_basis


def forecast(train, future):
    class WeekendGAM(PenalizedGAM):
        def _design(self, frame, fit=False):
            design = super()._design(frame, fit)
            hour, weekday, _, _ = calendar(frame)
            basis, penalty = periodic_basis(hour / 24, 12)
            interaction = basis * (weekday >= 5)[:, None]
            if fit:
                self.interaction_mean = interaction.mean(axis=0)
                self.penalty = block_diag(self.penalty, penalty)
            return np.column_stack([design, interaction - self.interaction_mean])

    fit, covs, weather = trailing_temperatures(train, future)
    model = WeekendGAM(Config(weather_columns=weather)).fit(fit)
    prediction = model.generate(None, covs)
    pd.testing.assert_series_equal(prediction.timestamp_utc, future.timestamp_utc)
    return prediction.y_pred_mw.to_numpy(), {**vars(model.config), "hour_by_weekend": True}
