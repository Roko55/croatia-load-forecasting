# Refit the development benchmark

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

Run the full four-year reproduction with:

```sh
python src/train.py --year 2019 2020 2021 2022 --verify
```

Add `--model` followed by one or more model IDs from
[the protocol](../configs/protocol.json) to select workflows. For the six
classical workflows, `requirements-training.txt` is sufficient.
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
and calendar baseline, and 0.05 MW for the pretrained workflows. For a
scoring-only check, run `python src/audit.py`.

The preparation and generation commands accept 2019–2022 only. Each fold
uses measured demand up to the preceding December 31 and supplies the target
year's calendar and ERA5 temperatures to the forecaster.
