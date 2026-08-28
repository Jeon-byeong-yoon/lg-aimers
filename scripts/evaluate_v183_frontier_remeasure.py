"""V183: redraw the frontier at V175, because the old map said the wrong thing.

V147 measured the frontier at V138 and concluded that pitcher-level information was close
to exhausted -- on 2024 the blend scored 903 against a pitcher-rate oracle's 991, an
88-point gap of which about 14 looked reachable. That map is why the project spent V152 to
V163 on components, weights and constants, and why V163 closed with "every known axis is
shut."

Then V169 and V175 took +14.65 from the calibration layer, out of a region the map had
called nearly empty. The map was not wrong about the pitcher *marginal* -- V165 confirmed
that marginal loses out of fold. It was wrong because it only ever measured marginals, and
the money was in a **within-pitcher interaction**.

Six experiments since have shut the calibration layer for good: new segments (V171, and
V176 where the placebos began passing), its eight parameters (V179, selection bias +4 to
+7), hierarchical shrinkage (V180, V181) and the correction algebra itself (V182). So
before guessing at the next axis, the map gets redrawn with the interaction structure the
old one lacked.

For each partition three numbers are reported on the 2024 fold.

  * **oracle** -- replace the prediction with the partition's true rate. The ceiling if
    that partition were all one knew.
  * **perfect correction** -- add each cell's true mean residual to the current blend.
    This is the ceiling for the calibration *mechanism*, which is the thing that has been
    paying.
  * **noise-corrected** -- the same, minus one variance per cell. V176 showed the raw
    figure crediting meaningless placebos with 549 points at 1,500 cells, so for fine
    partitions only the corrected column means anything, and even it is optimistic.

Nothing here is a candidate. It is a map, and the last one was believed for too long.
"""

import json
import sys
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


OUTPUT = Path("artifacts/v183_frontier_remeasure_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
V161_TERMS = [(["balls_before", "strikes_before"], 0.55, 500.0),
              (["pitcher_id", "balls_before", "strikes_before"], 0.25, 300.0),
              (["experience_bin"], 0.20, 2000.0)]
V175_TERMS = V161_TERMS + [(["pitcher_id", "batter_hand", "two_strike"], 0.80, 1500.0)]
PARTITIONS = {
    "count": ["count_state"],
    "experience": ["experience_bin"],
    "handedness": ["hand_matchup"],
    "pitcher": ["pitcher_id"],
    "batter": ["batter_id"],
    "pitcher x hand": ["pitcher_id", "batter_hand"],
    "pitcher x hand x 2K  (shipped)": ["pitcher_id", "batter_hand", "two_strike"],
    "pitcher x count": ["pitcher_id", "balls_before", "strikes_before"],
    "pitcher x hand x count": ["pitcher_id", "batter_hand", "count_state"],
    "batter x hand": ["batter_id", "pitcher_hand"],
    "pitcher x batter": ["pitcher_id", "batter_id"],
    "pitcher x inning": ["pitcher_id", "inning_bin"],
    "pitcher x month": ["pitcher_id", "game_month"],
    "placebo: pitcher x hand x parity": ["pitcher_id", "batter_hand", "day_parity"],
    "placebo: pitcher x parity": ["pitcher_id", "day_parity"],
}
P = 100000.0 / 0.25


def main():
    raw_frame = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    raw_frame = raw_frame.drop(columns=["row_id", "control_success"])
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v160 = joblib.load("artifacts/v160_reliability_scale_predictions.joblib")["predictions"]
    parts = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib"
                            )["forms"][20.0],
        "context": joblib.load("artifacts/v31_feature_removal_predictions.joblib"
                               )["no_matchup_hte"]["context"],
        "network": v160["network"][FEATURE_RELIABILITY],
        "catboost": joblib.load("artifacts/v153_projected_prior_predictions.joblib"
                                )["catboost"]["projected"],
        "factorization": v160["factorization"][FEATURE_RELIABILITY],
    }
    frame = raw_frame.copy()
    frame["experience_bin"] = pd.cut(frame["asof_pitcher_n"], EDGES,
                                     labels=LABELS).astype(str)
    frame["two_strike"] = (frame["strikes_before"] == 2).astype("int64")
    frame["count_state"] = (frame["balls_before"].astype(str) + "-"
                            + frame["strikes_before"].astype(str))
    frame["hand_matchup"] = (frame["pitcher_hand"].astype(str) + "-"
                             + frame["batter_hand"].astype(str))
    frame["inning_bin"] = np.minimum(frame["inning"], 10).astype(str)
    frame["day_parity"] = frame["game_dayofweek"] % 2
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
    blend = {year: sum(BASE[n] * parts[n][str(year)] for n in NAMES6) for year in YEARS}

    def calibrated(terms):
        scale = 1.0 / sum(w for _, w, _ in terms)
        pieces = []
        for year in YEARS:
            if year == 2022:
                pieces.append(np.clip(blend[year], 0, 1))
                continue
            residual = np.concatenate(
                [oof[str(h)]["target"].astype(float) - blend[h] for h in YEARS if h < year])
            train_f, valid_f = train_frames[year], valid_frames[year]
            value = blend[year] + residual.mean()
            for columns, weight, smoothing in terms:
                value = value + scale * weight * segment_correction(
                    train_f, residual, valid_f, columns, smoothing)
            pieces.append(np.clip(value, 0, 1))
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * term, 0, 1)

    mask = masks[2024]
    actual = target[mask]
    rate = actual.mean()

    def skill(prediction):
        return float(100000 * (1 - ((prediction - actual) ** 2).mean()
                               / (rate * (1 - rate))))

    v161 = calibrated(V161_TERMS)[mask]
    v175 = calibrated(V175_TERMS)[mask]
    audit = frame.loc[order].reset_index(drop=True).loc[mask]
    print(f"2024 fold, {mask.sum():,} rows, rate {rate:.6f}\n", flush=True)
    print(f"  V161 layer (pre-platoon)      skill {skill(v161):8.1f}", flush=True)
    print(f"  V175 layer (shipped)          skill {skill(v175):8.1f}   "
          f"the two accepted changes moved {skill(v175) - skill(v161):+.1f}", flush=True)
    print(f"  raw blend, no calibration     skill {skill(blend[2024]):8.1f}", flush=True)

    def key_of(columns):
        key = audit[columns[0]].astype(str)
        for column in columns[1:]:
            key = key + "|" + audit[column].astype(str)
        return key.to_numpy()

    print(f"\n{'partition':34s} {'cells':>7s} {'n/cell':>7s} "
          f"{'oracle':>8s} {'perfect':>9s} {'de-noised':>10s}", flush=True)
    rows = {}
    for label, columns in PARTITIONS.items():
        key = key_of(columns)
        table = pd.DataFrame({"y": actual, "p": v175, "k": key})
        grouped = table.groupby("k", sort=False)
        n = grouped.size().to_numpy().astype(float)
        truth = grouped["y"].mean()
        oracle = skill(table["k"].map(truth).to_numpy())
        offset = (grouped["y"].mean() - grouped["p"].mean())
        perfect = skill(v175 + table["k"].map(offset).to_numpy())
        residual = actual - v175
        rg = pd.DataFrame({"r": residual, "k": key}).groupby("k", sort=False)["r"]
        mean = rg.mean().to_numpy()
        var = np.nan_to_num(rg.var(ddof=1).to_numpy())
        denoised = float(P * (n * np.maximum(0.0, mean ** 2 - var / np.maximum(n, 1.0))
                              ).sum() / n.sum())
        rows[label] = {"cells": int(len(n)), "rows_per_cell": float(n.mean()),
                       "oracle_skill": oracle, "perfect_correction_skill": perfect,
                       "perfect_gain": perfect - skill(v175), "denoised_points": denoised}
        print(f"  {label:34s} {len(n):7d} {n.mean():7.0f} {oracle:8.0f} "
              f"{perfect - skill(v175):9.1f} {denoised:10.1f}", flush=True)

    shipped = rows["pitcher x hand x 2K  (shipped)"]["denoised_points"]
    placebo = max(rows[l]["denoised_points"] for l in rows if l.startswith("placebo"))
    print(f"\n  the de-noised column still credits the best placebo with {placebo:.0f} "
          f"points against the shipped term's {shipped:.0f}", flush=True)
    print("  -> at these cell sizes even the corrected figure is mostly inflation; "
          "read it only as a ranking, and only against the placebo line", flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V183_frontier_remeasure",
        "baseline": "V175 (Public 1067.8617513573)",
        "why": ("V147's map, measured at V138, said pitcher information was nearly "
                "exhausted and steered the project away from the region that later paid "
                "+14.65; it only ever measured marginals and the money was in a "
                "within-pitcher interaction"),
        "fold": {"year": 2024, "rows": int(mask.sum()), "rate": rate},
        "skills": {"raw_blend": skill(blend[2024]), "v161_layer": skill(v161),
                   "v175_layer": skill(v175),
                   "accepted_changes_moved": skill(v175) - skill(v161)},
        "partitions": rows,
        "caveat": ("V176 showed the noise correction failing at roughly 1,500 cells and "
                   "167 rows; the placebo rows are printed so the inflation is visible "
                   "rather than assumed away"),
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "in_fold_oracles_are_diagnostic_only": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
