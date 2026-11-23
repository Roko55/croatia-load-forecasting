"""LightGBM with calendar features and trailing temperatures."""
from ._features import calendar, trailing_temperatures


def forecast(train, future):
    from lightgbm import LGBMRegressor

    train, future, weather = trailing_temperatures(train, future, windows=(24, 72))
    model = LGBMRegressor(objective="regression_l1", n_estimators=320, num_leaves=31,
                          learning_rate=.05, random_state=42, n_jobs=1, deterministic=True,
                          force_col_wise=True, verbosity=-1)
    model.fit(calendar(train, weather), train.demand_mw)
    return model.predict(calendar(future, weather))
