"""Calendar, SARIMAX, additive, tree and feed-forward forecast workflows."""
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training_inputs import WEATHER
from statistical import Config, Features


def calendar(frame, weather):
    """Holiday flag supplied by the calendar input for every LOCAL hour, never merged at midnight here."""
    local = pd.DatetimeIndex(frame.timestamp_utc).tz_convert(ZoneInfo("Europe/Zagreb"))
    if not frame.is_holiday.isin([0, 1]).all():
        raise ValueError("is_holiday must be binary, all local hours marked by the calendar input")
    x = frame[list(weather) + ["is_holiday"]].astype(float).copy()
    for key, values, period in (("hour", local.hour, 24), ("week", local.dayofweek, 7),
                                ("annual", (local.dayofyear - 1) / np.where(local.is_leap_year, 366, 365), 1)):
        x[key + "_sin"] = np.sin(2 * np.pi * values / period)
        x[key + "_cos"] = np.cos(2 * np.pi * values / period)
    x["weekend"] = (local.dayofweek >= 5).astype(float)
    return x


def trailing_temperatures(train, future):
    """Use the current and previous 23 temperatures, including across Jan 1."""
    columns = tuple(f'{name}_avg24' for name in WEATHER)
    history = train[list(WEATHER)]
    combined = pd.concat([history, future[list(WEATHER)]], ignore_index=True)
    means = combined.rolling(24, min_periods=1).mean()
    train = train.assign(**dict(zip(columns, means.iloc[:len(train)].to_numpy().T)))
    future = future.assign(**dict(zip(columns, means.iloc[len(train):].to_numpy().T)))
    np.testing.assert_allclose(
        future.loc[0, list(columns)].to_numpy(dtype=float),
        np.vstack([history.to_numpy()[-23:], future.loc[0, list(WEATHER)].to_numpy(dtype=float)]).mean(axis=0),
        rtol=0, atol=1e-10,
    )
    assert 'demand_mw' not in future
    return train, future, (*WEATHER, *columns)


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


def calendar_baseline(train, future, half_life=4):
    from statistical import calendar

    def keys(frame):
        hour, dow, month, _ = calendar(frame)
        return pd.DataFrame({"month": month, "type": np.where(frame.is_holiday, 7, dow), "hour": hour})

    obs, pred = keys(train), keys(future)
    w = np.exp2(-np.arange(len(train)-1, -1, -1)/(half_life*365.25*24))
    obs["weight"], obs["weighted_load"] = w, w*train.demand_mw
    result = np.full(len(pred), np.average(train.demand_mw, weights=w))
    for group in (["hour"], ["type", "hour"], ["month", "type", "hour"]):
        totals = obs.groupby(group)[["weight", "weighted_load"]].sum()
        profiles = (totals.weighted_load / totals.weight).rename("value")
        joined = pred.join(profiles, on=group).value.to_numpy()
        result = np.where(np.isfinite(joined), joined, result)
    return result


def sarimax_weekend(train, future):
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


def gam_weekend(train, future):
    from scipy.linalg import block_diag
    from statistical import PenalizedGAM, calendar, periodic_basis

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


def lightgbm(train, future):
    from lightgbm import LGBMRegressor

    train, future, weather = trailing_temperatures(train, future)
    model = LGBMRegressor(objective="regression_l1", n_estimators=320, num_leaves=31,
                          learning_rate=.05, random_state=42, n_jobs=2, deterministic=True,
                          force_col_wise=True, verbosity=-1)
    model.fit(calendar(train, weather), train.demand_mw)
    return model.predict(calendar(future, weather))


def random_forest(train, future):
    from sklearn.ensemble import RandomForestRegressor

    train, future, weather = trailing_temperatures(train, future)
    x, future_x = (calendar(frame, weather) for frame in (train, future))
    model = RandomForestRegressor(n_estimators=200, max_samples=.65, max_depth=14,
                                  min_samples_leaf=24, max_features=.8, n_jobs=2, random_state=42)
    return model.fit(x, train.demand_mw).predict(future_x)


def mlp(train, future):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    train, future, weather = trailing_temperatures(train, future)
    x, future_x = (calendar(frame, weather) for frame in (train, future))
    y = train.demand_mw
    scaler = StandardScaler().fit(x)
    model = MLPRegressor(hidden_layer_sizes=(48, 24), alpha=.0001, batch_size=512, max_iter=1,
                         random_state=42, shuffle=True)
    center, scale = float(y.mean()), float(y.std())
    train_scaled = scaler.transform(x)
    for _ in range(20):
        model.partial_fit(train_scaled, (y-center)/scale)
    return model.predict(scaler.transform(future_x))*scale+center
