# Methodology

## Forecast design

At each January 1 UTC origin, each workflow generates a full year of hourly Croatian demand in MW. Calendar features use Europe/Zagreb local time. UTC is used for alignment and scoring, including all 24 leap-day hours in 2020.

| Development year | Available historical window | History hours | Scored hours |
|---|---|---:|---:|
| 2019 | 2006–2018 | 113,952 | 8,760 |
| 2020 | 2006–2019 | 122,712 | 8,784 |
| 2021 | 2006–2020 | 131,496 | 8,760 |
| 2022 | 2006–2021 | 140,256 | 8,760 |

The common forecast-year input is the calendar and observed ERA5 temperatures for Zagreb, Split, Rijeka and Osijek. There is no refresh with measured demand during the forecast year. Historical gaps use past-only forward filling for at most two consecutive hours. Every development target hour is source-valid.

The calendar baseline uses local hour, day type and month with recent observations weighted more heavily. Locally fitted models share the history split, with model-specific features, objectives and training budgets. Chronos-2 and TimesFM-3 use frozen pretrained checkpoints and the last 8,192 and 16,384 historical hours, respectively. Their external pretraining overlap with Croatian load is unknown. The experiment compares complete forecasting workflows. It cannot isolate the effect of model architecture alone.

## Scoring

Raw hourly mean absolute error (MAE) is the primary metric. For a year with $N$ one-hour observations, $y_t$ is observed load and $\hat y_t$ is forecast load at hour $t$, both in MW. Define the annual energy scale factor as $\alpha = \sum_t y_t / \sum_t \hat y_t$. The one-hour duration cancels in this ratio.

$$
\mathrm{MAE}_{\mathrm{raw}} = \frac{1}{N} \sum_{t=1}^{N} |y_t - \hat y_t|,
\qquad
\mathrm{MAE}_{\mathrm{shape}} = \frac{1}{N} \sum_{t=1}^{N} |y_t - \alpha \hat y_t|.
$$

Raw MAE reflects errors in both demand level and profile. Shape MAE holds annual energy fixed to compare how demand is distributed across the hours. The single multiplier preserves the order and timing of forecast values. It is computed separately for every model-year and used only in scoring.

RMSE, MAPE and signed annual energy bias use the raw forecasts. Peak magnitude error is the absolute difference between the two annual maxima after energy matching. Ramp MAE averages the absolute errors in consecutive hourly changes, after energy matching. Top-5% overlap is the share of observed top-load hours also among the highest forecast hours. Counts round up, with stable order resolving ties. Four-year means give each year equal weight. See the [development results](results.md) for the interpretation of these metrics.

The reported TCN profile is the hourly median of seeds 42–44. Other stochastic local workflows use the selected recipe's seed 42. See the [seed sensitivity table](../results/seed-sensitivity.csv). The four development years were used for model selection. Differences among the top models are descriptive.

The benchmark covers 2019–2022. The held-out year 2026 uses a 2006–2025 history window and will be evaluated when full-year weather and demand are available.

To regenerate the figures, install the optional plotting dependency
with `python -m pip install -r requirements-figures.txt` and run
`python src/figures.py`. Both scripts accept the same `--cache` argument.
