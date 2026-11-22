"""Conditional SARIMAX with calendar, weather and weekend-hour terms."""
import numpy as np

from training_inputs import WEATHER

from ._features import calendar
from ._statistical import Config, Features


def forecast(train, future):
    class WeekendFeatures(Features):
        def linear(self, frame, fit=False):
            base = super().linear(frame, fit=fit)
            cycles = calendar(frame, ())
            day = cycles[["hour_sin", "hour_cos"]].to_numpy() * cycles.weekend.to_numpy()[:, None]
            if fit:
                self.weekend_hour_mean = day.mean(axis=0)
            return np.column_stack([base, day - self.weekend_hour_mean])

    prediction, config = conditional_sarimax(train, future, features_class=WeekendFeatures)
    return prediction, {**config, "weekend_by_hour_harmonics": 1}


def conditional_sarimax(train, future, weather=WEATHER, features_class=Features):
    """Estimate the original SARIMAX order with conditional AR innovations."""
    from scipy.linalg import qr
    from scipy.optimize import minimize

    cfg = Config(weather_columns=weather)
    features = features_class(cfg)
    design = features.linear(train, fit=True)
    _, r, pivot = qr(design, mode='economic', pivoting=True)
    cols = np.sort(pivot[:np.linalg.matrix_rank(r)])
    x = design[:, cols]
    y = train.demand_mw.to_numpy()
    mean, scale = y.mean(), y.std()
    series = np.column_stack([x, (y - mean) / scale])
    lags = (0, 1, 24, 25)
    segments = [series[25 - lag:len(series) - lag] for lag in lags]
    # A single Gram matrix gives cross-products of all four filtered terms.
    stacked = np.concatenate(segments, axis=1)
    width = series.shape[1]
    gram = (stacked.T @ stacked).reshape(4, width, 4, width)
    del stacked

    def regression(ar):
        phi, seasonal = ar
        weights = np.array([1., -phi, -seasonal, phi * seasonal])
        cov = np.einsum('i,iajb,j->ab', weights, gram, weights)
        beta = np.linalg.solve(cov[:-1, :-1], cov[:-1, -1])
        rss = cov[-1, -1] - cov[-1, :-1] @ beta
        return rss / len(segments[0]), beta

    fits = [minimize(lambda ar: regression(ar)[0], point, bounds=[(-.995, .995)] * 2,
                     method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
            for point in ([0., 0.], [.7, .3])]
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not converged:
        raise RuntimeError('Conditional SARIMAX optimizer did not converge')
    fit = min(converged, key=lambda fit: fit.fun)
    mse, beta = regression(fit.x)
    phi, seasonal = fit.x
    residual = series[:, -1] - x @ beta
    direct = residual[25:] - phi * residual[24:-1] - seasonal * residual[1:-24] + phi * seasonal * residual[:-25]
    np.testing.assert_allclose(mse, np.mean(direct**2), rtol=1e-8, atol=1e-10)
    propagated = np.concatenate([residual[-25:], np.zeros(len(future))])
    for i in range(25, len(propagated)):
        propagated[i] = (phi * propagated[i - 1] + seasonal * propagated[i - 24]
                         - phi * seasonal * propagated[i - 25])
    predicted = (features.linear(future)[:, cols] @ beta + propagated[25:]) * scale + mean
    return predicted, {'configuration': vars(cfg), 'estimator': 'conditional Gaussian innovations',
                       'ar': fit.x.tolist(), 'fit_mse_standardized': mse,
                       'retained_exog_columns': cols.tolist()}
