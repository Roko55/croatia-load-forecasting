# Methodology

## Forecast design

At each January 1 UTC origin, each workflow generates a full year of hourly Croatian demand in MW. Calendar features use Europe/Zagreb local time. UTC is used for alignment and scoring, including all 24 leap-day hours in 2020 and 2024. At each origin, the training history grows by one year. This is an expanding-window version of rolling-origin evaluation. See [Tashman (2000)](https://doi.org/10.1016/S0169-2070(00)00065-0).

| Forecast year | Role | Available historical window | History hours | Scored hours |
|---|---|---|---:|---:|
| 2019 | Development | 2006–2018 | 113,952 | 8,760 |
| 2020 | Development | 2006–2019 | 122,712 | 8,784 |
| 2021 | Development | 2006–2020 | 131,496 | 8,760 |
| 2022 | Development | 2006–2021 | 140,256 | 8,760 |
| 2023 | Retrospective | 2006–2022 | 149,016 | 8,760 |
| 2024 | Retrospective | 2006–2023 | 157,776 | 8,784 |
| 2025 | Retrospective | 2006–2024 | 166,560 | 8,760 |

Model settings were selected using 2019–2022. The 2017 and 2018 target years supported an additional configuration sensitivity check for SARIMAX and TimesFM-3 during development. The 2023–2025 comparison fixes settings across years and rebuilds each forecast using its own history window. Earlier analyses had already inspected a subset of workflows on 2023–2025. The final held-out demand evaluation is reserved for 2026, with 2006–2025 history. See [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) for the effect of model selection on error estimates.

The common forecast-year inputs are the calendar and observed ERA5 temperatures for Zagreb, Split, Rijeka and Osijek. The scores measure accuracy conditional on realized weather, which would be unavailable at the start of an operational year-ahead forecast. There is no refresh with measured demand during the forecast year. Historical gaps use past-only forward filling for at most two consecutive hours. Every reported target hour is source-valid.

The calendar baseline uses local hour, day type and month with recent observations weighted more heavily. Locally fitted models share the history split, with model-specific features, objectives and training budgets. SARIMAX estimates a regression with daily seasonal AR errors by conditional Gaussian innovations. N-HiTS and TFT are fitted to historical load and exclude the lagged target from their input context. They forecast the annual profile in windows from covariates.

### Pretrained checkpoints

Chronos-2 and TimesFM-3 use frozen checkpoints and the last 8,192 and 16,384 historical hours, respectively. The Croatian demand passed at inference ends before each forecast origin. Pretraining was performed by the model developers. The public descriptions leave it unverified whether the training corpora contain Croatian load or overlapping derived series. See the [Chronos-2 model card](https://huggingface.co/amazon/chronos-2#training-data) and [TimesFM-3 model card](https://huggingface.co/google/timesfm-3.0-pytorch#data).

Amazon introduced Chronos-2 in October 2025. Google introduced TimesFM-3 on August 31, 2026, after every January 1 origin in this study, including the origin of the planned 2026 evaluation. Applying a checkpoint to an origin before its release is a retrospective simulation. Holding out 2026 demand in this project controls access during local fitting and inference. Pretraining independence for that year remains unverified. See the [Amazon announcement](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting) and [Google announcement](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/).

Chronos-2 generates the full 8,760 or 8,784 hours through the library's
long-horizon continuation, using its predicted median as feedback. This
extends beyond the checkpoint's recommended 1,024 steps. Its score applies
to this annual configuration. See the
[Chronos-2 pipeline](https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/src/chronos/chronos2/pipeline.py)
for the continuation rule. The experiment compares complete forecasting
workflows. It cannot isolate the effect of model architecture alone.

## Scoring

Raw hourly mean absolute error (MAE) is the primary metric. For a year with $N$ one-hour observations, $y_t$ is observed load and $\hat y_t$ is forecast load at hour $t$, both in MW. Define the annual energy scale factor as $\alpha = \sum_t y_t / \sum_t \hat y_t$. The one-hour duration cancels in this ratio.

$$
\mathrm{MAE}_{\mathrm{raw}} = \frac{1}{N} \sum_{t=1}^{N} |y_t - \hat y_t|,
\qquad
\mathrm{MAE}_{\mathrm{shape}} = \frac{1}{N} \sum_{t=1}^{N} |y_t - \alpha \hat y_t|.
$$

Raw MAE reflects errors in both demand level and profile. Shape MAE holds annual energy fixed to compare how demand is distributed across the hours. The single multiplier preserves the order and timing of forecast values. It is computed separately for every model-year and used only in scoring.

RMSE, MAPE and signed annual energy bias use the raw forecasts. Peak magnitude error is the absolute difference between the two annual maxima after energy matching. Ramp MAE averages the absolute errors in consecutive hourly changes, after energy matching. Top-5% overlap is the share of observed top-load hours also among the highest forecast hours. Counts round up, with stable order resolving ties. Period means give each year equal weight. The [development results](results.md) and [retrospective results](retrospective.md) report separate averages.

The reported TCN profile is the hourly median of seeds 42–44. Other stochastic local workflows use the selected recipe's seed 42. See the [seed sensitivity table](../results/seed-sensitivity.csv). Differences among the top models are descriptive. See [Hong, Pinson and Fan (2014)](https://doi.org/10.1016/j.ijforecast.2013.07.001) for an example of reporting a load-forecast benchmark with explicit inputs and scoring.

To regenerate the figures, install the optional plotting dependency
with `python -m pip install -r requirements-figures.txt` and run
`python src/figures.py`. Both scripts accept the same `--cache` argument.
