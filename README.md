# Croatian hourly electricity demand forecasting

Twelve forecasting workflows generate full-year hourly electricity demand profiles for Croatia, conditional on observed ERA5 weather. **2019–2022** was used for model development. **2023–2025** compares the 12 fixed configurations retrospectively. **2026** is reserved for the final evaluation.

[Run the benchmark](docs/reproduction.md) · [Read the methodology](docs/methodology.md) · [Results by year](docs/retrospective.md) · [Download hourly profiles](results/profiles/)

## Models and results

The table reports 2023–2025 means. Click a model name to open its Python implementation. Use the ID with `src/train.py --model`.

| Model code | `--model` ID | Raw hourly MAE (MW) | Equal-energy shape MAE (MW) |
|---|---|---:|---:|
| [TimesFM-3](src/models/timesfm3.py) | `timesfm3` | **94.98** | 94.25 |
| [LightGBM](src/models/lightgbm.py) | `lightgbm_thermal` | 110.11 | **90.44** |
| [MLP](src/models/mlp.py) | `ffn` | 111.13 | 91.06 |
| [Random Forest](src/models/random_forest.py) | `random_forest` | 111.44 | 93.51 |
| [GAM](src/models/gam.py) | `gam_thermal` | 118.06 | 105.10 |
| [LSTM](src/models/lstm.py) | `lstm` | 118.52 | 103.22 |
| [TCN](src/models/tcn.py) | `tcn_median` | 123.17 | 102.71 |
| [TFT](src/models/tft.py) | `tft` | 124.12 | 113.92 |
| [N-HiTS](src/models/nhits.py) | `nhits` | 128.52 | 117.27 |
| [Calendar baseline](src/models/calendar_baseline.py) | `calendar_recent` | 140.48 | 135.68 |
| [SARIMAX](src/models/sarimax.py) | `sarimax` | 160.18 | 151.37 |
| [Chronos-2](src/models/chronos2.py) | `chronos2` | 287.18 | 277.29 |

**Raw hourly MAE** scores the forecast as generated. **Shape MAE** scores the same hourly profile after scaling its annual energy to the observed total. This scaling is applied only during evaluation. Both report mean absolute hourly error in MW, with equal weight per year.

TimesFM-3 had the lowest mean raw MAE over 2023–2025. Among workflows without external pretraining, LightGBM had the lowest mean raw MAE. It also had the lowest mean shape MAE across all 12 workflows. The [methodology](docs/methodology.md#pretrained-checkpoints) explains checkpoint availability and the uncertainty about pretraining overlap.

[Per-year scores and observed-vs-forecast curves](docs/retrospective.md) · [Development results for 2019–2022](docs/results.md)

## Repository map

| Step | Where to look | What it does |
|---|---|---|
| Prepare inputs | [`src/training_inputs.py`](src/training_inputs.py) · [data sources](data/README.md) | Combines historical load, local calendar and ERA5. Writes a training file and a separate weather/calendar file per year. |
| Generate forecasts | [`src/train.py`](src/train.py) · [model files](src/models/) | Loads the two input files, runs the selected model and writes its hourly MW forecast. |
| Check results | [`src/audit.py`](src/audit.py) · [`results/`](results/) | Scores the archived forecasts against observed load and checks the published metrics. |
| Find settings | [Model settings](configs/model-settings.json) · [time splits](configs/protocol.json) · [pretrained checkpoints](configs/checkpoints.json) | Documents model options, forecast periods and the sources used to reproduce them. |

## Reproduce the scores

With Python 3.12 or later, run `python src/audit.py --period all` to recompute the published scores from the archived forecasts and source observations. See the [reproduction guide](docs/reproduction.md) to prepare inputs, rerun the models or rebuild figures.

The original project code and its documentation are [MIT-licensed](LICENSE). See the [data and checkpoint terms](data/README.md) for third-party materials and restrictions on TimesFM-3 outputs.
