# Reproduce the benchmark

For a scoring-only check, run `python src/audit.py --period all` with Python 3.12 or later. This uses the archived forecasts. The commands below generate forecasts from the source inputs.

Use a fresh Python 3.12 virtual environment. The full CPU setup includes the
six classical workflows, four fitted neural networks and two pretrained
workflows.

```sh
python -m pip install -r requirements-neural.txt
python src/training_inputs.py
python src/train.py --year 2022 --verify
```

The second command downloads the official ENTSO-E archives needed for
historical fitting, generates Croatian holiday flags and combines these with
the frozen ERA5 extract. It checks both input SHA-256 digests for each
development year. Use `--cache PATH` to reuse verified source files.
Inputs and forecasts are written under the ignored `work/` directory.

Run all models for the four development years:

```sh
python src/train.py --year 2019 2020 2021 2022 --verify
```

For the three retrospective years:

```sh
python src/training_inputs.py --year 2023 2024 2025
python src/train.py --year 2023 2024 2025 --verify
```

To run one model, take its ID from the [model table](../README.md#models-and-results).
For example:

```sh
python src/train.py --year 2022 --model lightgbm_thermal --verify
```

For the six classical workflows, `requirements-training.txt` is sufficient.
The runner executes models sequentially with at most two CPU threads.
TimesFM-3 selects the PyTorch math attention backend explicitly for stable
CPU inference across runs.

Chronos-2 and TimesFM-3 download their pinned checkpoints on first run
(about 1.7 GiB combined) and generate forecasts with frozen weights. They
have no local weight-fitting stage. Reuse verified downloads with
`--checkpoints PATH`, using the subdirectories `chronos2` and `timesfm3`
with `config.json` and `model.safetensors` inside each.

Verification compares every generated hour with its archived point forecast.
The maximum allowed difference is 0.005 MW for the locally fitted workflows
and calendar baseline, and 0.05 MW for the pretrained workflows. These
limits check numerical reproduction against the archive. The recorded
development reproduction verified all 48 profiles on Windows CPU. The
largest hourly difference was 0.00159 MW.

The full retrospective reproduction started from a clean checkout of
[commit `2e9e43c`](https://github.com/Roko55/croatia-load-forecasting/tree/2e9e43caaeb58be26e16702411ad56e94c882934).
The [execution report](../results/retrospective/reproduction-by-year.csv)
records the largest hourly difference and tolerance for each of the 36
profiles. The largest absolute difference across 315,648 hourly values
was $9.10 \times 10^{-13}$ MW. The
[provenance record](../results/retrospective/provenance.json)
records the runtime, exact commands and input manifest hashes, as well as
the code and driver used for the original archive.

The preparation and generation commands accept 2019–2025. Input preparation defaults to 2019–2022. Each fold
uses measured demand up to the preceding December 31 and supplies the target
year's calendar and ERA5 temperatures to the forecaster. N-HiTS and TFT
require at least one complete forecast window (168 and 336 hours,
respectively). The benchmark always supplies a full year.

## Automated checks

[GitHub Actions](https://github.com/Roko55/croatia-load-forecasting/actions)
runs the workbook boundary, input schema, DST, scoring and TimesFM-3
attention tests with fixtures and packaged weather. Use the reproduction
commands above for the full model runs and hour-by-hour comparisons.
