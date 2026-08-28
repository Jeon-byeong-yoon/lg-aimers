"""V178: the holdout killed V176 -- so check the holdout against two known answers.

V177 climbed the eight calibration parameters on 2024 months 3-7 and scored months 8-10:

    honest refit   2024 early +5.25   holdout -1.95
    greedy refit   2024 early +5.25   holdout -1.95
    V176 config    2024 early +3.23   holdout -1.93

Every refit loses the holdout, and both climbs drove `pitcher_count` to weight zero and
`count` smoothing to 75 -- the signature of fitting fold noise. The mirror term died the
same way, with the meaningless placebo actually scoring *better* on the holdout (+0.36)
than the real one (-1.79).

Before accepting that verdict, the instrument itself has to be calibrated, because a
holdout that is simply pessimistic would reject good changes as readily as bad ones. Two
changes have known answers: V169 returned +7.31 on the leaderboard and V175 returned
+7.32. If the holdout is sound it must score both positive. If it scores them negative,
the holdout is broken and V177's verdict means nothing.

This is the same move V131, V138 and V156 made -- calibrate the measuring device against
outcomes already in hand before trusting it on a new question.
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


OUTPUT = Path("artifacts/v178_holdout_instrument_metrics.json")
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
EDGES = [-1, 50, 150, 400, 1000, 2500, 6000, np.inf]
LABELS = ["0-50", "50-150", "150-400", "400-1k", "1k-2.5k", "2.5k-6k", "6k+"]
FEATURE_RELIABILITY = 300.0
DRIFT_SHRINKAGE, DRIFT_RELIABILITY, DRIFT_WEIGHT = 3.0, 150.0, 0.10
COUNT = ["balls_before", "strikes_before"]
PITCHER_COUNT = ["pitcher_id", "balls_before", "strikes_before"]
EXPERIENCE = ["experience_bin"]
PLATOON = ["pitcher_id", "batter_hand"]
PLATOON2K = ["pitcher_id", "batter_hand", "two_strike"]
LAYERS = {
    "V161": [(COUNT, 0.55, 500.0), (PITCHER_COUNT, 0.25, 300.0),
             (EXPERIENCE, 0.20, 2000.0)],
    "V169": [(COUNT, 0.55, 500.0), (PITCHER_COUNT, 0.25, 300.0),
             (EXPERIENCE, 0.20, 2000.0), (PLATOON, 0.20, 1000.0)],
    "V175": [(COUNT, 0.55, 500.0), (PITCHER_COUNT, 0.25, 300.0),
             (EXPERIENCE, 0.20, 2000.0), (PLATOON2K, 0.80, 1500.0)],
    "V177_refit": [(COUNT, 1.10, 75.0), (EXPERIENCE, 0.40, 300.0),
                   (PLATOON2K, 1.12, 900.0)],
}
KNOWN = {"V169": 7.3183, "V175": 7.3184}
SPLIT_MONTH = 8
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
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(add_training_inseason_features(
        raw_frame, shrinkage=DRIFT_SHRINKAGE,
        reliability_scale=DRIFT_RELIABILITY).loc[order], 1.0)
    validation_frame = make_validation_frame()
    season_of = validation_frame["season"].to_numpy()
    month_of = frame.loc[order, "game_month"].to_numpy()
    target = validation_frame["target"].to_numpy().astype(float)
    masks = {year: season_of == year for year in YEARS}
    early = masks[2024] & (month_of < SPLIT_MONTH)
    late = masks[2024] & (month_of >= SPLIT_MONTH)
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

    predictions = {label: calibrated(terms) for label, terms in LAYERS.items()}
    errors = {label: (p - target) ** 2 for label, p in predictions.items()}

    def delta(new, old, mask):
        return float(P * (errors[old][mask].mean() - errors[new][mask].mean()))

    steps = [("V169 over V161", "V169", "V161"),
             ("V175 over V169", "V175", "V169"),
             ("V177 refit over V175", "V177_refit", "V175")]
    print(f"holdout split: 2024 months <{SPLIT_MONTH} ({early.sum():,} rows) "
          f"vs >={SPLIT_MONTH} ({late.sum():,} rows)\n", flush=True)
    print(f"{'step':24s} {'2023':>8s} {'2024':>8s} {'early':>8s} {'HOLDOUT':>9s} "
          f"{'leaderboard':>12s}", flush=True)
    rows = {}
    for label, new, old in steps:
        row = {"2023": delta(new, old, masks[2023]), "2024": delta(new, old, masks[2024]),
               "2024_early": delta(new, old, early), "2024_late": delta(new, old, late)}
        rows[label] = row
        known = KNOWN.get(new)
        print(f"{label:24s} {row['2023']:8.2f} {row['2024']:8.2f} "
              f"{row['2024_early']:8.2f} {row['2024_late']:9.2f} "
              f"{('+%.2f' % known) if known else 'not submitted':>12s}", flush=True)

    verdict = (rows["V169 over V161"]["2024_late"] > 0
               and rows["V175 over V169"]["2024_late"] > 0)
    print(f"\nboth known winners positive on the holdout: {verdict}", flush=True)
    if verdict:
        print("  -> the holdout is a sound instrument, and V177's rejection of the "
              "joint refit stands", flush=True)
    else:
        print("  -> the holdout rejects changes that the leaderboard rewarded, so it is "
              "too pessimistic and V177's verdict cannot be trusted on its own", flush=True)

    # Month by month, for the two known winners: how consistent is the effect?
    print(f"\nmonth by month on the 2024 fold:", flush=True)
    months = sorted(set(month_of[masks[2024]].tolist()))
    header = "  " + " ".join(f"{m:>7d}" for m in months)
    print(f"{'':24s}{header}", flush=True)
    monthly = {}
    for label, new, old in steps:
        values = [delta(new, old, masks[2024] & (month_of == m)) for m in months]
        monthly[label] = dict(zip((str(m) for m in months), values))
        won = sum(1 for v in values if v > 0)
        print(f"{label:24s}  " + " ".join(f"{v:7.2f}" for v in values)
              + f"   {won}/{len(values)}", flush=True)

    OUTPUT.write_text(json.dumps({
        "experiment": "V178_holdout_instrument_check",
        "question": ("V177's holdout rejected every joint refit; before accepting that, "
                     "the holdout is scored against two changes whose leaderboard answers "
                     "are known -- V169 returned +7.31 and V175 returned +7.32"),
        "split_month": SPLIT_MONTH,
        "rows": {"2024_early": int(early.sum()), "2024_late": int(late.sum())},
        "steps": rows,
        "known_leaderboard": KNOWN,
        "monthly": monthly,
        "holdout_is_sound": bool(verdict),
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "row_independent_segments": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
