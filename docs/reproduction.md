# Reproduce the benchmark

For a scoring-only check, run `python src/audit.py --period all` with Python 3.12 or later. This uses the archived forecasts. The commands below generate forecasts from the source inputs.

Clone the repository, then create and activate a Python 3.12 virtual environment.

```sh
git clone https://github.com/Roko55/croatia-load-forecasting.git
cd croatia-load-forecasting
```

Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux (shell)

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

Use the full CPU dependency set to generate forecasts.
The pinned PyTorch CPU build is available for Windows and Linux. The
scoring-only command also works on macOS with Python 3.12 or later.

```sh
python -m pip install -r requirements-neural.txt
python src/training_inputs.py
python src/train.py --year 2022
```

`src/training_inputs.py` downloads the official ENTSO-E archives needed for
historical fitting, generates Croatian holiday flags and combines these with
the frozen ERA5 extract. It checks both input SHA-256 digests for each
requested year. Use `--cache PATH` to reuse verified source files.
Inputs and forecasts are written under the ignored `work/` directory.

Run all models for the four development years:

```sh
python src/train.py --year 2019 2020 2021 2022
```

For the three retrospective years:

```sh
python src/training_inputs.py --year 2023 2024 2025
python src/train.py --year 2023 2024 2025
```

To run one model, take its ID from the [model table](../README.md#models-and-results).
For example:

```sh
python src/train.py --year 2022 --model lightgbm_thermal --verify
```

For the six classical workflows, `requirements-training.txt` is sufficient.
The runner executes models sequentially with at most two CPU threads.
TimesFM-3 selects the PyTorch math attention backend explicitly for stable
CPU inference. Its native attention can require substantially more RAM than the other models.

Chronos-2 and TimesFM-3 download their pinned checkpoints on first run
(about 1.7 GiB combined) and generate forecasts with frozen weights. They
have no local weight-fitting stage. Reuse verified downloads with
`--checkpoints PATH`, using the subdirectories `chronos2` and `timesfm3`
with `config.json` and `model.safetensors` inside each.

The archive contains hourly predictions. The locally fitted workflows are
fitted again each time you regenerate a profile.

Add `--verify` to compare every generated hour with its archived point forecast.
The maximum allowed difference is 0.005 MW for locally fitted workflows
and the calendar baseline, and 0.05 MW for pretrained workflows. CPU and
numerical library differences can change fitted weights and predictions.
The [forecast reproduction report](../results/reproduction-by-year.csv)
records the hours checked, maximum difference and tolerance for each
model and year. Run `python src/audit.py --check-reproduction` to check its
coverage and tolerances. Run `python src/audit.py --period all` to
recalculate published metrics directly from the pinned archives. See
[provenance](../results/provenance.json) for the generation code revision,
runtime, and manifest checksums.

The preparation and generation commands accept 2019–2025. Input preparation defaults to 2019–2022. Each fold
uses measured demand up to the preceding December 31 and supplies the target
year's calendar and ERA5 temperatures to the forecaster. N-HiTS and TFT
require at least one complete forecast window (168 and 336 hours,
respectively). The benchmark always supplies a full year.

## Automated checks

[GitHub Actions](https://github.com/Roko55/croatia-load-forecasting/actions)
runs the workbook boundary, input schema, DST, scoring and prior-year
load feature tests with fixtures and packaged weather. It checks the
configuration selection and the recorded forecast reproduction report.
Use the reproduction commands above to refit the models and compare
the generated forecasts hour by hour.
