# Data and archived predictions

## Observed electricity demand

The audit retrieves the 2019–2025 **Monthly Hourly Load Values** files from [ENTSO-E Power Statistics](https://www.entsoe.eu/data/power-stats/). It selects the Croatian rows (`CountryCode=HR`), interprets `DateUTC` as UTC and uses `Value` in MW. The download endpoints and SHA-256 values are pinned in [entsoe-sources.json](entsoe-sources.json).

The archive download cache is ignored by Git. To use already downloaded copies of the exact archives, place them in one directory and run `python src/audit.py --period all --cache /path/to/archives`. Each file is verified before use. Third-party inputs remain subject to their source terms.

## Weather, calendar and model weights

The research forecasts used [Copernicus ERA5 hourly single-level data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) and Croatian holiday and local-time features. The frozen input hashes appear in [the protocol](../configs/protocol.json). The metric audit only requires observed load and archived forecasts.

The two external checkpoint references are [Chronos-2 (Apache 2.0)](https://huggingface.co/amazon/chronos-2) and
[TimesFM-3 (TimesFM Non-Commercial License v1.0)](https://huggingface.co/google/timesfm-3.0-pytorch#license). Revisions and weight checksums are recorded in the [checkpoint manifest](../configs/checkpoints.json).
The [TimesFM Non-Commercial License](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE) governs use of the pinned model and restricts commercial or production use of its outputs. The project license covers its original software and documentation.

## Frozen forecasts

The seven compressed tables in [results/profiles](../results/profiles) contain UTC timestamps and model-generated MW forecasts, exported at round-trip floating-point precision. The 2019–2022 archives include the three constituent TCN forecasts, whose hourly median is scored. The 2023–2025 archives store that median directly. The observed series is read from the ENTSO-E files at run time.

## Forecast inputs

`src/training_inputs.py` reads the ENTSO-E legacy workbook for 2006–2015,
the bridge workbook from 31 December 2015 at 23:00 UTC through 2018, and
the annual CSVs from 2019. The legacy workbook's coverage ends at Zagreb's
local New Year, so the bridge supplies the last UTC hour of 2015. The
[ENTSO-E Power Statistics source](https://www.entsoe.eu/data/power-stats/)
changes to Transparency Platform aggregates from 2019. On the 7,295
overlapping Croatian hours from January through October 2019, the annual
CSV averages 33.01 MW below the bridge workbook and the mean absolute
difference is 34.35 MW. The benchmark uses the annual CSV from 2019 onward. Links and checksums are in
[training-sources.json](training-sources.json) and
[entsoe-sources.json](entsoe-sources.json). The command fills historical
gaps using up to two preceding hours of carry-forward. The scoring inputs
use source observations directly.

The two ERA5 Parquet extracts contain UTC timestamps and the four
cities' 2-metre temperatures for 2006–2025, with the original numeric values preserved.
The temperatures were obtained via the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).
The upstream ERA5 product is available from the
[Copernicus Climate Data Store](https://doi.org/10.24381/cds.adbb2d47).

Attribution for both extracts is “Contains modified Copernicus Climate Change
Service information (2006–2025).”
The weather extract is provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
