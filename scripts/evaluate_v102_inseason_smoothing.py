"""V102: refit the two smoothing constants inside the in-season reconstruction.

`SHRINKAGE = 50` and `RELIABILITY_SCALE = 150` were picked a priori when the
reconstruction was written in V92 and were never revisited. They then propagated
unchanged through V93 (into the Form model's features) and V96 (into the drift
term). This is the same class of staleness V96 fixed, and it sits in the family
that produced +50 leaderboard points.

The constants are not cosmetic. The success-rate delta's 2024 mean moves from
-0.0167 at shrinkage 20 to -0.0017 at shrinkage 200 — the shrinkage decides how
much of a pitcher's current-season record the model is allowed to believe.

Shrinkage changes the features, so the Form model is retrained for each value.
The reliability scale only enters one feature and the post-hoc term; for the
post-hoc term it is recomputed from `ins_log_n_pitcher`, so it is swept for free.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_v96_v93_weight_reoptimization import blend_and_calibrate
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, feature_names
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v102_inseason_smoothing_metrics.json")
PREDICTIONS = Path("artifacts/v102_inseason_smoothing_predictions.joblib")
FORM_CONFIG = {"max_leaf_nodes": 15, "min_samples_leaf": 200,
               "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500}
SHRINKAGES = (20.0, 50.0, 100.0, 200.0)
RELIABILITY_SCALES = (50.0, 150.0, 400.0)
DRIFT_SCALES = (0.10, 0.15, 0.20)
BASELINE = (50.0, 150.0, 0.15)
W_FORM, W_CONTEXT = 0.52, 0.21
INSEASON = feature_names()
MIN_GAIN = 1.06e-5
RATIO = 0.932


def points(gain):
    return float(gain * 100000.0 / 0.25)


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])

    blocks, forms, terms = {}, {}, {}
    for shrinkage in SHRINKAGES:
        block = add_training_inseason_features(raw_frame, shrinkage=shrinkage)
        blocks[shrinkage] = block[INSEASON]
        rows = block.loc[order]
        delta = np.nan_to_num(
            rows["dlt_asof_pitcher_success_rate"].to_numpy(dtype=float), nan=0.0)
        inside_n = np.expm1(rows["ins_log_n_pitcher"].to_numpy(dtype=float))
        terms[shrinkage] = {
            scale: delta * (inside_n / (inside_n + scale))
            for scale in RELIABILITY_SCALES
        }
        print(f"features built for shrinkage={shrinkage}", flush=True)

    for shrinkage in SHRINKAGES:
        forms[shrinkage] = {}
        for year in YEARS:
            train_mask = raw_frame["season"] < year
            valid_mask = raw_frame["season"] == year
            prior = float(y.loc[train_mask].mean())
            frame = select_v2_features(add_row_features(form_raw, prior))
            frame = add_trackman_features(frame, trackman)
            frame = pd.concat([frame, blocks[shrinkage]], axis=1)
            columns = v31_form_columns(frame)
            model, model_columns = hist_gbdt_pipeline(frame[columns])
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v for k, v in FORM_CONFIG.items()
            })
            model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
            forms[shrinkage][str(year)] = model.predict_proba(
                frame.loc[valid_mask, model_columns])[:, 1]
            print(f"shrinkage={shrinkage} year={year} form trained", flush=True)

    blends = {
        shrinkage: blend_and_calibrate(
            W_FORM, W_CONTEXT, oof, logistic, forms[shrinkage], context, raw_frame)
        for shrinkage in SHRINKAGES
    }
    validation_frame = make_validation_frame()
    baseline = np.clip(
        blends[BASELINE[0]] + BASELINE[2] * terms[BASELINE[0]][BASELINE[1]], 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    candidates, development = {}, {}
    for shrinkage in SHRINKAGES:
        for scale in RELIABILITY_SCALES:
            for drift in DRIFT_SCALES:
                if (shrinkage, scale, drift) == BASELINE:
                    continue
                label = f"k{shrinkage:.0f}_rel{scale:.0f}_w{drift:.2f}"
                candidates[label] = np.clip(
                    blends[shrinkage] + drift * terms[shrinkage][scale], 0, 1)
                development[label] = development_metrics(reference, candidates[label])
        print(f"evaluated shrinkage={shrinkage}", flush=True)

    def passes(label):
        r = development[label]
        return (r["season_gain_development"]["2022"] > -1e-5
                and r["season_gain_development"]["2023"] > -1e-5
                and r["gain_2024_mar_aug"] >= MIN_GAIN
                and r["gain_2024_jul_aug"] > 0
                and r["monthly_block_win_rate"] >= 0.75)

    eligible = [label for label in development if passes(label)]

    # Plateau on all three axes: neighbours in the grid must also pass. Three
    # candidates have now failed the sealed window after scraping past on a
    # knife-edge development margin, so a lone passing point is not trusted.
    def key_of(label):
        k, rel, w = label.split("_")
        return (float(k[1:]), float(rel[3:]), float(w[1:]))

    axes = (SHRINKAGES, RELIABILITY_SCALES, DRIFT_SCALES)
    all_labels = set(development) | {f"k{BASELINE[0]:.0f}_rel{BASELINE[1]:.0f}_w{BASELINE[2]:.2f}"}

    def label_of(key):
        return f"k{key[0]:.0f}_rel{key[1]:.0f}_w{key[2]:.2f}"

    def neighbours(key):
        out = []
        for axis, values in enumerate(axes):
            index = values.index(key[axis])
            for step in (-1, 1):
                if 0 <= index + step < len(values):
                    candidate = list(key)
                    candidate[axis] = values[index + step]
                    name = label_of(tuple(candidate))
                    if name in all_labels:
                        out.append(name)
        return out

    plateau = [
        label for label in eligible
        if all(n in eligible or n == label_of(BASELINE) for n in neighbours(key_of(label)))
    ]
    pool = plateau or eligible
    promoted = max(pool, key=lambda l: development[l]["gain_2024_mar_aug"], default=None)
    final_confirmation = None
    if promoted is not None:
        final_confirmation = {"candidate": promoted,
                              "comparison": compare_candidate(reference, candidates[promoted])}

    output = {
        "experiment": "V102_inseason_smoothing_constants",
        "baseline": "V96 (shrinkage 50, reliability scale 150, drift 0.15)",
        "baseline_public_score": 962.8787800874,
        "rationale": (
            "SHRINKAGE=50 and RELIABILITY_SCALE=150 were chosen a priori in V92 and never "
            "refitted, yet they propagated into V93's features and V96's drift term."
        ),
        "shrinkages": list(SHRINKAGES),
        "reliability_scales": list(RELIABILITY_SCALES),
        "drift_scales": list(DRIFT_SCALES),
        "blend_weights": {"v17": round(1 - W_FORM - W_CONTEXT, 2),
                          "form": W_FORM, "context": W_CONTEXT},
        "selection_rule": "pass all criteria and all existing grid neighbours must pass too",
        "min_gain_for_submission": MIN_GAIN,
        "local_to_leaderboard_ratio": RATIO,
        "development_results": development,
        "eligible_candidates": sorted(eligible),
        "plateau_candidates": sorted(plateau),
        "promoted_candidate": promoted,
        "final_confirmation_inspected": promoted is not None,
        "final_confirmation": final_confirmation,
        "compliance": {
            "official_data_only": True,
            "test_csv_read": False,
            "test_row_aggregation_used": False,
            "constants_frozen_at_training_time": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"forms": forms, "baseline": baseline}, PREDICTIONS, compress=3)

    print("\ntop 20 by 2024 development gain (vs V96):")
    print(f"{'candidate':>22} {'2022':>8} {'2023':>8} {'2024pt':>8} {'LB est':>7} {'blocks':>8} {'jul_aug':>12} {'plateau':>8}")
    for label, r in sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"])[:20]:
        sg = r["season_gain_development"]
        print(f"{label:>22} {points(sg['2022']):8.1f} {points(sg['2023']):8.1f} "
              f"{points(r['gain_2024_mar_aug']):8.1f} "
              f"{points(r['gain_2024_mar_aug'])*RATIO:7.1f} "
              f"{r['monthly_block_win_rate']:8.2%} {r['gain_2024_jul_aug']:12.3e} "
              f"{'YES' if label in plateau else '-':>8}")
    print(f"\neligible={len(eligible)}  plateau={len(plateau)}\npromoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
