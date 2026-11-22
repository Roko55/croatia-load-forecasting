"""Fit a development workflow and generate its complete annual load profile."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit import ROOT, digest, predictions, read_json, require, source_file
import classical
from training_inputs import YEARS, validate, write_frame

CLASSICAL = {
    "calendar_recent": classical.calendar_baseline,
    "sarimax": classical.sarimax_weekend,
    "gam_thermal": classical.gam_weekend,
    "lightgbm_thermal": classical.lightgbm,
    "random_forest": classical.random_forest,
    "ffn": classical.mlp,
}
PRETRAINED = ("chronos2", "timesfm3")


def checkpoint(method, cache):
    from pretrained import checkpoint_spec

    spec = checkpoint_spec()[method]
    base = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}"
    folder = cache / method
    for file, sha in (("config.json", spec["config_sha256"]), ("model.safetensors", spec["sha256"])):
        source_file({"file": file, "download_url": f"{base}/{file}", "sha256": sha}, folder)
    return folder


def generate(method, train, future, checkpoints):
    if method in CLASSICAL:
        result = CLASSICAL[method](train, future)
        return result[0] if isinstance(result, tuple) else result

    if method == "chronos2":
        from pretrained import chronos

        return chronos(train, future, checkpoint(method, checkpoints))[0]

    import torch
    import neural

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if method == "timesfm3":
        from pretrained import timesfm_context

        return timesfm_context(torch, train, future, checkpoint(method, checkpoints))[0]
    if method == "tcn_median":
        trials = [neural.tcn_control(torch, train, future, seed)[0] for seed in (42, 43, 44)]
        return np.median(np.stack(trials), axis=0)
    if method == "lstm":
        return neural.lstm(torch, train, future, 42)[0]
    if method == "nhits":
        engine, _, settings = neural.direct(method, 168, train, future, 42)
        return neural.centered_days(engine, train, future, settings)
    if method == "tft":
        return neural.direct(method, 336, train, future, 42)[1]
    raise ValueError(f"Unknown workflow: {method}")


def train_one(year, method, inputs, outputs, checkpoints, verify, protocol):
    fold = protocol["development"][str(year)]
    directory = inputs / str(year)
    paths = directory / "training.parquet", directory / "covariates.parquet"
    for path, field in zip(paths, ("training_sha256", "covariate_sha256"), strict=True):
        require(path.is_file() and digest(path) == fold[field],
                f"{path.name} is missing or differs from the frozen input. Run src/training_inputs.py.")
    history, future = map(pd.read_parquet, paths)
    validate(history, future, year)
    print(f"{year} {method}: generating {len(future):,} hours", flush=True)
    with threadpool_limits(limits=2):
        result = np.asarray(generate(method, history, future, checkpoints), dtype=float)
    require(result.shape == (len(future),) and np.isfinite(result).all() and (result > 0).all(),
            f"{year} {method}: forecast must have one positive finite value per hour.")
    if verify:
        expected = predictions(year, fold, protocol["models"])[method]
        error = float(np.max(np.abs(result - expected)))
        limit = .05 if method in PRETRAINED else .005
        require(error <= limit, f"{year} {method}: maximum absolute difference {error:.9g} MW exceeds {limit} MW.")
        print(f"{year} {method}: matches archived profile, max difference {error:.9g} MW", flush=True)
    output = outputs / str(year) / f"{method}.parquet"
    write_frame(output, pd.DataFrame({"timestamp_utc": future.timestamp_utc, "prediction_mw": result}))


def main():
    protocol = read_json(ROOT / "configs" / "protocol.json")
    methods = tuple(protocol["models"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", choices=YEARS, required=True, type=int, nargs="+")
    parser.add_argument("--model", choices=methods, nargs="+", default=methods)
    parser.add_argument("--inputs", type=Path, default=ROOT / "work" / "inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "predictions")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "data" / "cache" / "checkpoints")
    parser.add_argument("--verify", action="store_true", help="Compare every generated hour with the archived profile")
    args = parser.parse_args()
    if len(args.year) == len(args.model) == 1:
        train_one(args.year[0], args.model[0], args.inputs, args.output, args.checkpoints, args.verify, protocol)
        return
    for year in args.year:
        for method in args.model:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--year", str(year), "--model", method,
                       "--inputs", str(args.inputs), "--output", str(args.output),
                       "--checkpoints", str(args.checkpoints)]
            if args.verify:
                command.append("--verify")
            process = subprocess.run(command, check=False)
            if process.returncode:
                raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
