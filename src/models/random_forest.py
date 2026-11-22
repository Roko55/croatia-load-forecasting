"""Random Forest with calendar features and trailing temperatures."""
from ._features import calendar, trailing_temperatures


def forecast(train, future):
    from sklearn.ensemble import RandomForestRegressor

    train, future, weather = trailing_temperatures(train, future)
    x, future_x = (calendar(frame, weather) for frame in (train, future))
    model = RandomForestRegressor(n_estimators=200, max_samples=.65, max_depth=14,
                                  min_samples_leaf=24, max_features=.8, n_jobs=2, random_state=42)
    return model.fit(x, train.demand_mw).predict(future_x)
