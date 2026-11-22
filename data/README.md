# Data and archived predictions

## Observed electricity demand

The audit retrieves the 2019–2022 **Monthly Hourly Load Values** files from [ENTSO-E Power Statistics](https://www.entsoe.eu/data/power-stats/). It selects the Croatian rows (`CountryCode=HR`), interprets `DateUTC` as UTC and uses `Value` in MW. The download endpoints and SHA-256 values are pinned in [entsoe-sources.json](entsoe-sources.json).

The archive download cache is ignored by Git. To use already downloaded copies of the exact archives, place them in one directory and run `python src/audit.py --cache /path/to/archives`. Each file is verified before use. Third-party inputs remain subject to their source terms.

## Weather, calendar and model weights

The research forecasts used [Copernicus ERA5 hourly single-level data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) and Croatian holiday and local-time features. The frozen development input hashes appear in [the protocol](../configs/protocol.json). The metric audit only requires observed load and archived forecasts.

The two external checkpoint references are [Chronos-2](https://huggingface.co/amazon/chronos-2) and [TimesFM-3](https://huggingface.co/google/timesfm-3.0-pytorch). Revisions and weight checksums are recorded in [the model settings](../configs/model-settings.json).

## Frozen forecasts

The four compressed tables in [results/profiles](../results/profiles) contain UTC timestamps and model-generated MW forecasts, exported at round-trip floating-point precision. They include the three constituent TCN forecast vectors. The audit combines these hourly. The observed series is read from the ENTSO-E files at run time.
