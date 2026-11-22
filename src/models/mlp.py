"""Feed-forward multilayer perceptron with calendar and temperature features."""
from ._features import calendar, trailing_temperatures


def forecast(train, future):
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
