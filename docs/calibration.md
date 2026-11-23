# Configuration selection evidence

The benchmark reports one selected configuration per workflow. This supplement records the configuration settings and the 2019–2022 scores used by the selection rule. See the [methodology](methodology.md#forecast-design) for thresholds and ranking.

| File | Contents |
|---|---|
| [Candidate settings](../results/calibration/candidates.json) | Configuration options, the selected configuration and its identifier in the benchmark. Options describe changes to the reference workflow. |
| [Development scores](../results/calibration/development-scores.csv) | Raw and equal-energy shape MAE for each configuration and development year. |
| [Seed sensitivity](../results/calibration/seed-sensitivity.csv) | Individual scores for seeds 43 and 44 used to check qualifying stochastic configurations. Seed 42 appears in the development table. TCN development scores use its hourly median over all three seeds. |

`reference` is the configuration used for comparison on the evaluation host. For workflows with a matched reference run, `published_reference` is the archived score from the [reference snapshot](https://github.com/Roko55/croatia-load-forecasting/tree/c18d5e508af19479f25b8e776842fe18f85b9b0e) and serves as a second comparison. Both references are included for every model to keep the same table structure. If no candidate qualified, the benchmark retains the published reference.

The search included prior-year load features for LSTM, forecast feedback for N-HiTS and TFT, and shorter forecast windows for Chronos-2. The scores describe the annual forecasting strategies in the candidate settings.

Run `python src/selection_audit.py` to recalculate the selection from these tables and compare it with the development scores in the benchmark. The script checks the recorded scores and the decision rule. See the [reproduction guide](reproduction.md) for regenerating the selected forecasts.
