# Retrospective results for 2023–2025

At each January 1 UTC origin, the 12 fixed configurations use demand available through the preceding December and the target year's calendar and observed ERA5 weather. Earlier work inspected some outcomes from these years. The final held-out demand evaluation is reserved for 2026. See the [methodology](methodology.md) for the split, data availability and metric definitions.

## Hourly error by year

All MAEs are in MW. Raw MAE scores the original forecast. Shape MAE scores after matching the observed annual energy. Bold marks the lowest error in each column. Models are ordered by their mean raw MAE, with equal weight per year.

| Model | 2023 raw | 2023 shape | 2024 raw | 2024 shape | 2025 raw | 2025 shape |
|---|---:|---:|---:|---:|---:|---:|
| TimesFM-3 | 86.58 | 86.55 | **90.47** | **88.50** | **107.89** | 107.71 |
| LightGBM | **78.39** | **70.19** | 110.34 | 91.23 | 141.60 | 109.92 |
| MLP | 82.13 | 73.33 | 112.97 | 94.36 | 138.28 | **105.48** |
| Random Forest | 81.69 | 75.30 | 111.55 | 94.89 | 141.09 | 110.34 |
| GAM | 91.00 | 86.39 | 118.98 | 107.11 | 144.19 | 121.79 |
| LSTM | 92.14 | 91.39 | 114.22 | 104.07 | 149.19 | 114.20 |
| TCN | 86.64 | 83.94 | 131.59 | 109.24 | 151.29 | 114.95 |
| TFT | 98.59 | 96.97 | 118.19 | 115.33 | 155.59 | 129.44 |
| N-HiTS | 101.24 | 98.49 | 135.60 | 120.15 | 148.72 | 133.18 |
| Calendar baseline | 117.22 | 117.24 | 153.43 | 149.10 | 150.79 | 140.69 |
| SARIMAX | 132.16 | 129.69 | 177.82 | 170.52 | 170.56 | 153.91 |
| Chronos-2 | 340.37 | 332.82 | 274.10 | 277.23 | 247.07 | 221.81 |

![Raw and equal-energy MAE by model and year, with period means](../results/retrospective/comparison.png)

TimesFM-3 had the lowest mean raw MAE (94.98 MW) and the highest overlap with the observed top 5% of hours (69.22%). LightGBM had the lowest mean shape MAE (90.44 MW), followed by the MLP (91.06 MW). In 2025, the lowest raw and shape errors came from TimesFM-3 and the MLP, respectively. These rankings describe the three forecast years.

## Profiles and downloads

Weekly means show the annual curves against observed demand for every workflow. Scores use the original hourly values.

| Target year | Observed vs forecast | Hourly forecasts |
|---|---|---|
| 2023 | [All 12 curves](../results/figures/profiles-2023.png) | [CSV.gz](../results/profiles/2023.csv.gz) |
| 2024 | [All 12 curves](../results/figures/profiles-2024.png) | [CSV.gz](../results/profiles/2024.csv.gz) |
| 2025 | [All 12 curves](../results/figures/profiles-2025.png) | [CSV.gz](../results/profiles/2025.csv.gz) |

[All metrics by year](../results/retrospective/by-year.csv) · [Period means](../results/retrospective/summary.csv) · [Run provenance](../results/retrospective/provenance.json) · [Reproduction commands](reproduction.md)
