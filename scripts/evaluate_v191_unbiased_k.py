"""V191: V190's k table is contaminated -- every averaged set contained the lucky seed.

V190 scored the average of k against seeds it does not contain, which fixed one bias, and
left another. The sets were nested:

    k=1  {42}                            2024  +9.73
    k=3  {42, 1004, 2024}   shipped      2024  +8.86
    k=7  {42, ..., 2718}                 2024  +7.01

Every set contains seed 42, and V188 showed seed 42 to be the luckiest of five draws on the
2024 fold. At k=1 it carries the whole average, at k=7 one seventh -- so the 2024 column
falls with k not because averaging gets worse but because **seed 42's luck is being diluted**.
That luck is a property of these particular fold fits and will not be in the deployed model,
so the table reads exactly backwards for the decision it is meant to inform.

The k=1 row is the cleanest statement of the problem: a single seed beats the held-out seeds
by +9.73 on 2024, which is not a fact about averaging at all. It is seed 42's draw.

The fix is to average over *which* seeds go inside. For each k, many random subsets of the
nine are drawn, each scored against the seeds it excludes, and the readings pooled. Seed 42
then appears in the inside set exactly as often as any other, and what is left is the effect
of averaging.

Expected shape, if the mechanism is what V188 and V189 said it is: the gain against a fresh
single draw rises with k and saturates, because averaging k independent draws removes
`1 - 1/k` of the lottery variance. If instead the curve is flat or falling once seed 42 is
de-weighted, then V189's gain was seed luck after all and that is worth knowing.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_v137_context_slot_replacement import NAMES6
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v191_unbiased_k_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
         (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
         (["experience_bin"], 0.20, 2000.0),
         (["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
FEATURE_SHRINKAGE, FEATURE_RELIABILITY = 20.0, 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
SEEDS = (42, 1004, 2024, 777, 999, 13, 314, 2718, 65537)
MAX_SUBSETS = 20
P = 100000.0 / 0.25


def main():
    raw_frame = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw_frame = raw_frame.drop(columns=["row_id", "control_success"])
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    v164 = joblib.load("artifacts/v164_seed_averaging_predictions.joblib")
    v190 = joblib.load("artifacts/v190_form_lottery_and_k_predictions.joblib")
    networks = dict(v164["networks"])
    networks.update(v190["networks"])
    assert set(SEEDS) <= set(networks), sorted(set(SEEDS) - set(networks))
    forms = dict(v190["forms"])
    forms.update(joblib.load(
        "artifacts/v192_more_form_seeds_predictions.joblib")["forms"])
    form_seeds = tuple(sorted(forms))
    # Both defaults are what V189 ships, so a curve on one component holds the other
    # at the deployed configuration rather than at a bare single draw.
    fixed = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                            )["forms"][FEATURE_SHRINKAGE],
        "network": {str(y): np.mean([networks[s][str(y)] for s in (42, 1004, 2024)],
                                    axis=0) for y in YEARS},
        "context": joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                               )["no_matchup_hte"]["context"],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
        "factorization": v160["factorization"][FEATURE_RELIABILITY],
    }
    frame = raw_frame.copy()
    frame["experience_bin"] = pd.cut(frame["asof_pitcher_n"], EDGES,
                                     labels=LABELS).astype(str)
    frame["two_strike"] = (frame["strikes_before"] == 2).astype("int64")
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    masks = {year: season_of == year for year in YEARS}
    train_frames = {y: frame.loc[np.concatenate(
        [oof[str(h)]["row_index"] for h in YEARS if h < y])]
        for y in YEARS if y != 2022}
    valid_frames = {y: frame.loc[oof[str(y)]["row_index"]] for y in YEARS}
    scale = 1.0 / sum(w for _, w, _ in TERMS)

    def pipeline(network=None, form=None):
        parts = dict(fixed)
        if network is not None:
            parts["network"] = network
        if form is not None:
            parts["form"] = form
        blend = {y: sum(BASE[n] * parts[n][str(y)] for n in NAMES6) for y in YEARS}
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
            value = blend[year] + residual.mean()
            for columns, weight, smoothing in TERMS:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    def points(candidate, base_error):
        error = (candidate - target) ** 2
        return [float(P * (base_error[masks[y]].mean() - error[masks[y]].mean()))
                for y in YEARS]

    def mean_of(store, seeds):
        return {str(y): np.mean([store[s][str(y)] for s in seeds], axis=0)
                for y in YEARS}

    generator = np.random.default_rng(20260829)

    def curve(store, seeds, label, kwarg):
        base_error = {s: (pipeline(**{kwarg: store[s]}) - target) ** 2 for s in seeds}
        print(f"\n{label}: average of k against the seeds it excludes, "
              f"pooled over which seeds go inside", flush=True)
        print(f"  {'k':>3s} {'subsets':>8s} {'2022':>8s} {'2023':>8s} {'2024':>8s} "
              f"{'sd(2024)':>9s}", flush=True)
        table = {}
        for k in range(1, len(seeds)):
            every = list(combinations(seeds, k))
            if len(every) > MAX_SUBSETS:
                picked = [every[i] for i in generator.choice(
                    len(every), MAX_SUBSETS, replace=False)]
            else:
                picked = every
            readings = []
            for inside in picked:
                avg = (store[inside[0]] if k == 1 else mean_of(store, inside))
                candidate = pipeline(**{kwarg: avg})
                for s in seeds:
                    if s in inside:
                        continue
                    readings.append(points(candidate, base_error[s]))
            block = np.array(readings)
            mean = block.mean(axis=0)
            table[k] = {"subsets": len(picked), "readings": len(readings),
                        "expected": mean.tolist(),
                        "sd": block.std(axis=0, ddof=1).tolist()}
            print(f"  {k:3d} {len(picked):8d} {mean[0]:+8.2f} {mean[1]:+8.2f} "
                  f"{mean[2]:+8.2f} {block.std(axis=0, ddof=1)[2]:9.2f}", flush=True)
        return table

    network_table = curve(networks, SEEDS, "embedding network", "network")
    form_table = curve(forms, form_seeds, "Form", "form")

    best_network = max(network_table, key=lambda k: network_table[k]["expected"][2])
    print(f"\nnetwork: best k by expected 2024 = {best_network} "
          f"({network_table[best_network]['expected'][2]:+.2f}); "
          f"shipped k=3 is {network_table[3]['expected'][2]:+.2f}", flush=True)
    if form_table:
        best_form = max(form_table, key=lambda k: form_table[k]["expected"][2])
        print(f"form: best k by expected 2024 = {best_form} "
              f"({form_table[best_form]['expected'][2]:+.2f})", flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V191_unbiased_k",
        "baseline": "V189 (Public 1071.4549348488)",
        "bias_fixed": ("V190's averaged sets were nested and every one contained seed 42, "
                       "the luckiest draw of five on the 2024 fold, so its 2024 column "
                       "fell with k because that luck was being diluted rather than "
                       "because averaging got worse; here the inside set is pooled over "
                       "random subsets so every seed appears equally often"),
        "seeds": list(SEEDS),
        "max_subsets_per_k": MAX_SUBSETS,
        "network_k_table": network_table,
        "form_k_table": form_table,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
