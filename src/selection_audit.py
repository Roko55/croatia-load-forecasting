"""Verify the selected workflows from the recorded development scores."""
import csv
from math import fsum, isfinite

from audit import ROOT, digest, read_json, require

CALIBRATION = ROOT / "results" / "calibration"
METRICS = ("mae_mw", "shape_mae_mw")


def read_scores(path, identity, metrics):
    rows = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[column] for column in identity)
            values = tuple(float(row[column]) for column in metrics)
            require(key not in rows, f"Duplicate score in {path.name}: {key}")
            require(all(isfinite(value) and value >= 0 for value in values),
                    f"Invalid score in {path.name}: {key}")
            rows[key] = values
    return rows


def ranking(candidate, baseline, policy):
    """Rank qualifying configurations with raw error before shape error."""
    change = [fsum(values) / fsum(reference) - 1
              for values, reference in zip(zip(*candidate), zip(*baseline), strict=True)]
    better = [sum(value < previous for value, previous in zip(values, reference, strict=True))
              for values, reference in zip(zip(*candidate), zip(*baseline), strict=True)]
    minimum_years = policy["minimum_years_improved"]
    for index, reduction in enumerate((policy["raw_mae_reduction"], policy["shape_mae_reduction"])):
        if (change[index] <= -reduction and better[index] >= minimum_years
                and change[1 - index] <= policy["maximum_other_metric_increase"]):
            return index, change[index], change[1 - index]
    return None


def audit_selection():
    catalog = read_json(CALIBRATION / "candidates.json")
    provenance = read_json(ROOT / "results" / "provenance.json")
    calibration = provenance["calibration_evidence"]
    for filename, sha256 in calibration["sha256"].items():
        require(digest(CALIBRATION / filename) == sha256, f"Unexpected calibration file: {filename}")
    protocol = read_json(ROOT / "configs" / "protocol.json")
    years = tuple(protocol["development"])
    require(years == tuple(map(str, provenance["selection_years"])) == tuple(map(str, catalog["development_years"])),
            "Selection years differ")
    policy = provenance["selection_rule"]
    models = catalog["models"]
    require(set(models) == set(protocol["models"]), "Candidate models differ from the benchmark")
    keys = ("model", "configuration", "year")
    scores = read_scores(CALIBRATION / "development-scores.csv", keys, METRICS)
    seed_scores = read_scores(CALIBRATION / "seed-sensitivity.csv", (*keys, "seed"), METRICS)
    benchmark = read_scores(ROOT / "results" / "by-year.csv", ("model", "year"), METRICS)
    expected_scores = {(model, name, year) for model, spec in models.items()
                       for name in (*spec["configurations"], "published_reference") for year in years}
    require(set(scores) == expected_scores, "Missing or unexpected development score")
    reference_count = len(models) * len(years)
    development_count = len(scores) - reference_count
    require(calibration["development_model_year_scores"] == development_count
            and calibration["archived_reference_scores"] == reference_count
            and calibration["additional_seed_scores"] == len(seed_scores),
            "Incorrect calibration evidence counts")
    required_seeds = set()
    winners = []
    for model, spec in models.items():
        def annual(configuration):
            return [scores[model, configuration, year] for year in years]

        reference = annual("reference")
        published = annual("published_reference")
        if not spec["matched_reference_rerun"]:
            require(all(abs(x - y) <= 1e-8 for row, base in zip(reference, published, strict=True)
                        for x, y in zip(row, base, strict=True)), f"Reference scores differ for {model}")
        ranked = []
        for name in spec["configurations"]:
            if name == "reference":
                continue
            values = annual(name)
            order = ranking(values, reference, policy)
            if order is None or ranking(values, published, policy) is None:
                continue
            if spec["three_seed_check"]:
                def raw_across_seeds(configuration):
                    pairs = [(model, configuration, year, seed) for year in years for seed in ("43", "44")]
                    required_seeds.update(pairs)
                    require(all(key in seed_scores for key in pairs), f"Missing seed scores for {model}/{configuration}")
                    return fsum([row[0] for row in annual(configuration)]
                                + [seed_scores[key][0] for key in pairs])

                allowed = policy["stochastic_mean_raw_mae_maximum_increase"] if order[0] else 0
                if raw_across_seeds(name) > raw_across_seeds("reference") * (1 + allowed):
                    continue
            ranked.append((order, name))
        order, winner = min(ranked) if ranked else (None, "reference")
        require(winner == spec["selected_configuration"], f"Selection differs for {model}")
        require(spec["benchmark_recipe"] == protocol["models"][model]["recipe"],
                f"Published workflow identifier differs for {model}")
        selected = published if winner == "reference" else annual(winner)
        for year, recorded in zip(years, selected, strict=True):
            require(all(abs(a - b) <= 1e-8 for a, b in zip(recorded, benchmark[model, year], strict=True)),
                    f"Selected scores differ from the benchmark for {model}/{year}")
        winners.append((model, winner, ("raw_mae", "shape_mae")[order[0]] if order else "reference"))
    require(set(seed_scores) == required_seeds, "Unexpected seed score")
    return winners, development_count


if __name__ == "__main__":
    results, checked = audit_selection()
    for model, configuration, criterion in results:
        print(f"{model}: {configuration} ({criterion})")
    print(f"PASS: {checked} development scores and {len(results)} model selections.")
