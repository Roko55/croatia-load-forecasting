"""Seasonal features and the penalized additive load model."""
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Config:
    weather_columns: tuple = ("temperature_c",)
    order: tuple = (1, 0, 0)
    seasonal_order: tuple = (1, 0, 0, 24)
    maxiter: int = 100
    spline_intervals: int = 8
    smoothing: float = 10.0
    ridge: float = 1e-6
    annual_harmonics: int = 3
    use_exog: bool = True
    trend: str = "n"
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True


def checked(frame, target=False):
    """Strict UTC hourly grid; reject naive times and nonfinite training truth."""
    f = frame.copy(deep=True)
    raw = pd.to_datetime(f["timestamp_utc"], errors="raise")
    if raw.dt.tz is None:
        raise ValueError("timestamp_utc must be timezone aware")
    if any(t.utcoffset().total_seconds() != 0 for t in raw):
        raise ValueError("timestamp_utc must explicitly use UTC")
    f["timestamp_utc"] = raw.dt.tz_convert("UTC")
    if len(f) == 0 or f.timestamp_utc.duplicated().any():
        raise ValueError("empty or duplicate timestamps")
    steps = f.timestamp_utc.diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError("timestamps must be increasing contiguous UTC hours")
    h = f["is_holiday"].to_numpy(dtype=float)
    if not np.isin(h, [0, 1]).all():
        raise ValueError("is_holiday must be audited binary flags")
    local = f.timestamp_utc.dt.tz_convert(ZoneInfo("Europe/Zagreb"))
    supplied_calendar = {"hour_local": local.dt.hour, "dayofweek_local": local.dt.dayofweek,
        "dayofyear_local": local.dt.dayofyear, "month_local": local.dt.month, "year_local": local.dt.year,
        "utc_offset_minutes": [int(t.utcoffset().total_seconds() / 60) for t in local],
        "is_weekend": (local.dt.dayofweek >= 5).astype(int)}
    for name, expected in supplied_calendar.items():
        if name in f and not np.array_equal(f[name].to_numpy(), np.asarray(expected)):
            raise ValueError("supplied local calendar disagrees with UTC timestamp: " + name)
    if pd.DataFrame({"day": local.dt.date, "h": h}).groupby("day").h.nunique().max() > 1:
        raise ValueError("holiday flag must cover every local hour of the day")
    if target:
        y = f["demand_mw"].to_numpy(dtype=float)
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("training demand must be finite and nonnegative")
    return f.reset_index(drop=True)


def calendar(f):
    t = f.timestamp_utc.dt.tz_convert(ZoneInfo("Europe/Zagreb"))
    # Fixed leap-calendar phase: March has the same phase in leap/nonleap years.
    doy = np.array([pd.Timestamp(2000, x.month, x.day).dayofyear - 1 for x in t])
    hour = t.dt.hour.to_numpy(dtype=float)
    return hour, t.dt.dayofweek.to_numpy(), t.dt.month.to_numpy(), (doy + hour / 24) / 366


class Base:
    model_id = "abstract"

    def __init__(self, config=None):
        self.config = config if isinstance(config, Config) else Config(**(config or {}))
        cols = self.config.weather_columns
        if not isinstance(cols, (tuple, list)) or len(set(cols)) != len(cols):
            raise ValueError("weather_columns must be a unique list/tuple")
        if any(not isinstance(k, str) or k in ("timestamp_utc", "is_holiday") or
               any(word in k.lower() for word in ("truth", "target", "lag", "load", "demand", "y_true")) for k in cols):
            raise ValueError("weather feature cannot contain load/truth/lag or reserved names")
        self.fitted = False

    def _fit_input(self, train_frame, validation_frame, config):
        if config is not None:
            raise ValueError("configure at construction; fit-time override is unsupported")
        if validation_frame is not None:
            raise ValueError("adapter never trains on validation; use the annual evaluation procedure")
        f = checked(train_frame, target=True)
        self.train_end_utc = f.timestamp_utc.iloc[-1]
        self.origin_utc = self.train_end_utc + pd.Timedelta(hours=1)
        self.train_timestamps = f.timestamp_utc.copy()
        return f

    def _future(self, context_frame, future_covariates, config, allow_gap=False):
        if not self.fitted:
            raise RuntimeError("fit required")
        if config is not None:
            raise ValueError("generation cannot change fitted config")
        # Supplied context must end within the fitted history.
        if context_frame is not None and len(context_frame):
            c = checked(context_frame)
            if c.timestamp_utc.max() > self.train_end_utc:
                raise ValueError("context extends beyond fitted origin")
        f = checked(future_covariates)
        if f.timestamp_utc.iloc[0] < self.origin_utc or (not allow_gap and f.timestamp_utc.iloc[0] != self.origin_utc):
            raise ValueError("horizon must start one hour after training origin")
        return f

    def _output(self, f, values):
        values = np.asarray(values, dtype=float)
        if hasattr(self, "model_diagnostics"):
            finite = values[np.isfinite(values)]
            self.model_diagnostics["generation"] = {
                "start_utc": f.timestamp_utc.iloc[0].isoformat(), "end_utc": f.timestamp_utc.iloc[-1].isoformat(),
                "requested_hours": len(f), "returned_values": int(values.size),
                "nonfinite_count": int((~np.isfinite(values)).sum()), "negative_count": int((values < 0).sum()),
                "finite_min_mw": float(finite.min()) if finite.size else None,
                "finite_max_mw": float(finite.max()) if finite.size else None,
                "clipping_applied": False, "clipped_count": 0, "max_clipping_correction_mw": 0.0,
                "status": "GENERATED" if values.shape == (len(f),) and np.isfinite(values).all() and (values >= 0).all() else "REJECTED_OUTPUT"}
        if values.shape != (len(f),) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("negative, nonfinite or incorrectly shaped prediction")
        return pd.DataFrame({"timestamp_utc": f.timestamp_utc,
                             "y_pred_mw": values, "model_id": self.model_id,
                             "origin_utc": self.origin_utc})

    def generate_planning(self, future_covariates, config=None):
        """Direct calendar/weather scenario generation for models without AR state."""
        f = self._future(None, future_covariates, config, allow_gap=True)
        return self._predict_frame(f)


class Features:
    def __init__(self, config):
        self.config = config

    def weather(self, f, fit=False):
        if not self.config.weather_columns:
            return np.empty((len(f), 0))
        x = f[list(self.config.weather_columns)].to_numpy(dtype=float)
        if np.isinf(x).any():
            raise ValueError("infinite weather")
        if fit:
            if np.isnan(x).all(axis=0).any():
                raise ValueError("all-missing training weather")
            self.medians = np.nanmedian(x, axis=0)
        return np.where(np.isnan(x), self.medians, x)

    def linear(self, f, fit=False):
        """Build the regression matrix from calendar harmonics and weather."""
        if not 0 <= self.config.annual_harmonics <= 6:
            raise ValueError("annual_harmonics must be in [0, 6]")
        if fit and self.config.annual_harmonics and len(f) < 8760:
            raise ValueError("annual SARIMAX harmonics require at least 8760 training hours; use 0 for short fixtures")
        hour, dow, _, year = calendar(f)
        terms = [f.is_holiday.to_numpy(dtype=float)]
        terms.extend((dow == k).astype(float) for k in range(1, 7))
        for phase, harmonics in [(hour / 24, 3), ((dow + hour / 24) / 7, 3), (year, self.config.annual_harmonics)]:
            for k in range(1, harmonics + 1):
                terms.extend([np.sin(2 * np.pi * k * phase), np.cos(2 * np.pi * k * phase)])
        x = np.column_stack(terms + [self.weather(f, fit=fit)])
        if fit:
            # Reuse training statistics at forecast time and drop constant features.
            self.mean = x.mean(axis=0)
            self.scale = x.std(axis=0)
            self.keep = self.scale > 1e-10
        return np.column_stack([np.ones(len(f)), (x[:, self.keep] - self.mean[self.keep]) / self.scale[self.keep]])


def periodic_basis(phase, n):
    """Uniform cubic B-splines with wrapped coefficients: exactly cyclic."""
    knots = np.arange(-3, n + 4, dtype=float) / n
    raw = BSpline.design_matrix(np.asarray(phase) % 1, knots, 3).toarray()
    out = np.zeros((len(phase), n))
    # Tie coefficients across the cycle boundary and penalize wrapped curvature.
    for j in range(raw.shape[1]):
        out[:, j % n] += raw[:, j]
    d = np.eye(n) - 2 * np.roll(np.eye(n), 1, axis=1) + np.roll(np.eye(n), 2, axis=1)
    return out, d.T @ d


class PenalizedGAM(Base):
    """Gaussian identity-link P-spline GAM, fitted by penalized least squares.

    Smooth cyclic hour/year + smooth weather terms + weekday/holiday effects.
    Objective: ||y-Xb||^2 + lambda sum ||D2 b_term||^2 + epsilon ||b||^2.
    """
    model_id = "penalized_additive_gam_v1"

    def _design(self, f, fit=False):
        """Combine hour, year, temperature and weekday effects into an additive design."""
        hour, dow, _, year = calendar(f)
        blocks, penalties = [], []
        for phase, n in [(hour / 24, 12), (year, 12)]:
            b, p = periodic_basis(phase, n)
            blocks.append(b)
            penalties.append(p)
        weather = self.features.weather(f, fit=fit)
        if fit:
            # Place knots using training quantiles and reuse the same basis for prediction.
            self.knots = []
            for j in range(weather.shape[1]):
                breaks = np.unique(np.quantile(weather[:, j], np.linspace(0, 1, self.config.spline_intervals + 1)))
                if len(breaks) < 2:
                    self.knots.append(None)
                else:
                    self.knots.append(np.r_[np.repeat(breaks[0], 3), breaks, np.repeat(breaks[-1], 3)])
        for j, knots in enumerate(self.knots):
            if knots is None:
                continue
            # Evaluate temperatures outside the training range at the nearest boundary.
            b = BSpline.design_matrix(np.clip(weather[:, j], knots[3], knots[-4]), knots, 3).toarray()
            d = np.diff(np.eye(b.shape[1]), n=2, axis=0)
            blocks.append(b)
            penalties.append(d.T @ d)
        linear = np.column_stack([f.is_holiday.to_numpy(dtype=float)] + [(dow == k).astype(float) for k in range(1, 7)])
        blocks.append(linear)
        penalties.append(np.zeros((7, 7)))
        x = np.column_stack(blocks)
        if fit:
            self.basis_means = x.mean(axis=0)
            self.penalty = block_diag(np.zeros((1, 1)), *penalties)
        return np.column_stack([np.ones(len(f)), x - self.basis_means])

    def fit(self, train_frame, validation_frame=None, config=None):
        f = self._fit_input(train_frame, validation_frame, config)
        if not np.isfinite([self.config.smoothing, self.config.ridge]).all() or self.config.smoothing <= 0 or self.config.ridge <= 0 or not 4 <= self.config.spline_intervals <= 20:
            raise ValueError("GAM requires positive penalties and 4..20 spline intervals")
        self.features = Features(self.config)
        x = self._design(f, fit=True)
        reg = self.config.smoothing * self.penalty + self.config.ridge * np.eye(x.shape[1])
        # Penalize curvature within smooth terms and leave the intercept unpenalized.
        reg[0, 0] = 0
        with threadpool_limits(limits=2):
            self.coef = np.linalg.solve(x.T @ x + reg, x.T @ f.demand_mw.to_numpy(dtype=float))
        self.fitted = True
        return self

    def generate(self, context_frame, future_covariates, config=None):
        f = self._future(context_frame, future_covariates, config)
        return self._predict_frame(f)

    def _predict_frame(self, f):
        with threadpool_limits(limits=2):
            y = self._design(f) @ self.coef
        return self._output(f, y)
