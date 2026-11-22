# Croatian hourly electricity demand forecasting

A comparison of **12 workflows** that generate full-year hourly Croatian electricity demand profiles. The development benchmark covers **2019–2022** and uses observed ERA5 weather. Actual load from each target year is used for scoring. The held-out year 2026 is reserved for final evaluation.

[Run the benchmark](docs/reproduction.md) · [Read the methodology](docs/methodology.md) · [View results by year](results/by-year.csv) · [Download hourly profiles](results/profiles/)

## Models and results

Click a model name to open its Python implementation. Each file is in [src/models](src/models/) and exposes a `forecast` function. Use the ID in the second column with `src/train.py --model`.

| Model code | `--model` ID | Hourly MAE (MW) | Shape MAE (MW) |
|---|---|---:|---:|
| [MLP](src/models/mlp.py) | `ffn` | **87.11** | 74.33 |
| [LightGBM](src/models/lightgbm.py) | `lightgbm_thermal` | 88.94 | **72.87** |
| [Random Forest](src/models/random_forest.py) | `random_forest` | 89.11 | 75.11 |
| [TCN](src/models/tcn.py) | `tcn_median` | 95.66 | 84.65 |
| [LSTM](src/models/lstm.py) | `lstm` | 97.40 | 83.31 |
| [GAM](src/models/gam.py) | `gam_thermal` | 102.19 | 88.49 |
| [N-HiTS](src/models/nhits.py) | `nhits` | 102.28 | 93.93 |
| [TFT](src/models/tft.py) | `tft` | 103.47 | 86.94 |
| [TimesFM-3](src/models/timesfm3.py) | `timesfm3` | 110.88 | 89.12 |
| [Calendar baseline](src/models/calendar_baseline.py) | `calendar_recent` | 115.25 | 104.71 |
| [SARIMAX](src/models/sarimax.py) | `sarimax` | 132.12 | 120.69 |
| [Chronos-2](src/models/chronos2.py) | `chronos2` | 246.24 | 205.13 |

Scores are averaged with equal weight per year. Shape MAE compares profiles after matching annual energy at scoring time. Chronos-2 and TimesFM-3 use frozen pretrained weights. [Full metric tables](results/summary.csv) · [Seed sensitivity](results/seed-sensitivity.csv)

## Repository map

| Step | Where to look | What it does |
|---|---|---|
| Prepare inputs | [`src/training_inputs.py`](src/training_inputs.py) · [data sources](data/README.md) | Combines historical load, local calendar and ERA5. Writes a training file and a separate weather/calendar file per year. |
| Generate forecasts | [`src/train.py`](src/train.py) · [model files](src/models/) | Loads the two input files, runs the selected model and writes its hourly MW forecast. |
| Check results | [`src/audit.py`](src/audit.py) · [`results/`](results/) | Scores the archived forecasts against observed load and checks the published metrics. |
| Find settings | [Model settings](configs/model-settings.json) · [time splits](configs/protocol.json) · [pretrained checkpoints](configs/checkpoints.json) | Documents model options, forecast periods and the sources used to reproduce them. |

## Compare the profiles

Weekly means by model for [2019](results/figures/profiles-2019.png), [2020](results/figures/profiles-2020.png), [2021](results/figures/profiles-2021.png) and [2022](results/figures/profiles-2022.png). Metrics use the original hourly values.

![Weekly mean observed and predicted load in 2022 for all 12 models](results/figures/profiles-2022.png)

To check the published metrics with Python 3.12+, run `python src/audit.py`. For preparing data and generating forecasts, see the [reproduction guide](docs/reproduction.md).
