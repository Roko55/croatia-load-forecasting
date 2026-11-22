# Development results

The benchmark compares 12 full-year forecasts for 2019–2022. See [how the metrics are calculated](methodology.md#scoring), the [annual results](../results/by-year.csv) and the [four-year means](../results/summary.csv).

## Winners by year

| Year | Lowest raw hourly MAE | Lowest equal-energy shape MAE |
|---|---:|---:|
| 2019 | Random Forest · 84.62 MW | LightGBM and Random Forest · 72.75 MW |
| 2020 | LightGBM · 82.01 MW | LightGBM · 76.20 MW |
| 2021 | MLP · 83.58 MW | LightGBM · 67.07 MW |
| 2022 | TimesFM-3 · 88.95 MW | MLP · 75.11 MW |

Values are rounded to two decimals. These years were used for model selection. MLP led LightGBM by 1.83 MW on four-year raw MAE, while MLP's mean MAE across individual seeds ranged from 86.09 to 90.52 MW. Treat the ranking as descriptive. [Seed sensitivity](../results/seed-sensitivity.csv)

## What the scores tell us

| Question | Four-year leader | Average result |
|---|---|---:|
| Smallest error in the forecasts as generated | MLP | 87.11 MW raw MAE |
| Closest hourly profile at equal annual energy | LightGBM | 72.87 MW shape MAE |
| Closest annual peak magnitude at equal annual energy | MLP | 101.76 MW absolute peak error |
| Closest average hour-to-hour changes at equal annual energy | Calendar baseline | 21.95 MW ramp MAE |
| Best identification of the highest-load hours | TimesFM-3 | 67.63% top-5% overlap |

MLP reduced raw MAE by 24.4% compared with the calendar baseline (115.25 MW). Ramp MAE averages errors across all hourly changes. Annual peak error measures magnitude, and top-5% overlap measures agreement on which hours had high demand.

## Hourly example

The seven-day window is centred on the highest observed load in 2022. The top panel shows the raw forecasts, and the bottom panel shows them after scaling each **entire year** to the observed annual energy. The figure illustrates the transformation on actual hourly curves. The metrics in the tables are calculated over all hours.

![Hourly MLP and LightGBM forecasts before and after annual-energy scaling, alongside observed load](../results/figures/peak-week-2022.png)

[All-model weekly plots](../results/figures/) · [Hourly forecast files](../results/profiles/)

## Scope

The forecasts use observed ERA5 weather. This evaluates how the workflows reproduce load under a known weather path. The final 2026 comparison and the effect of the demand profiles on H2RES planning remain open.
