"""V94a: extend the in-season reconstruction to Context, and drop `season`.

Two cheap structural hypotheses are measured separately and in combination,
against the V93 baseline (Public 951.1067955895).

H1 — Context HGB has never seen the current-season reconstruction. V93 only fed
it to the Form layer (32% of the blend); Context is another 13%.

H2 — `season` is a raw numeric feature in every tree model. Trees cannot
extrapolate, so `season=2025` falls into the same leaf as 2024 and the model
structurally predicts 2025 as "the 2024 league". V93 produced direct evidence
that this matters: removing V92's global drift term collapsed the monthly block
win rate from 100% to 70%, i.e. the trees never internalised the league shift.
If `season` is removed, the level is carried entirely by the calibration shift
and the drift term, which *can* track it.

The V17 tree layer (55%) keeps `season` here; that is the heavier V94b step.
Candidate selection uses 2022, 2023 and 2024 March-August only.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v38_lr_grid import BASE_CONFIG, v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS, calibrated_prediction
from evaluate_v88_transfer_validation import compare_candidate, make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v94a_context_inseason_metrics.json")
PREDICTIONS = Path("artifacts/v94a_context_inseason_predictions.joblib")
FORM_CONFIG = {**BASE_CONFIG, "learning_rate": 0.03, "max_iter": 500}
MATCHUP_HTE = {
    "hte_pitcher_batter_100", "hte_pitcher_batter_500",
    "hte_pitcher_batter_log_count",
}
DRIFT_WEIGHT = 0.20
INSEASON = feature_names()
# Local 2024 development gain that is worth a submission (~+4 leaderboard points
# at the measured cumulative ratio of 0.94).
MIN_GAIN = 1.06e-5
RATIO = 0.944


def points(gain):
    return float(gain * 100000.0 / 0.25)


def fit_predict(frame, columns, config, train_mask, valid_mask, y):
    candidate = frame[columns]
    model, model_columns = hist_gbdt_pipeline(candidate)
    if config:
        model.set_params(**{
            f"histgradientboostingclassifier__{key}": value
            for key, value in config.items()
        })
    model.fit(candidate.loc[train_mask, model_columns], y.loc[train_mask])
    return model.predict_proba(candidate.loc[valid_mask, model_columns])[:, 1]


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    inseason_full = add_training_inseason_features(raw_frame)
    inseason_block = inseason_full[INSEASON]

    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)
    context_trackman = prepare_context_trackman(trackman)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v93_form = joblib.load("artifacts/v93_inseason_form_model_predictions.joblib")
    v93_form = v93_form["new_form"]["with_2019_nan"]
    v93_context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    v93_context = v93_context["no_matchup_hte"]["context"]

    ctx = {name: {} for name in ("inseason", "noseason", "inseason_noseason")}
    form_noseason = {}
    counts = {}
    for year in YEARS:
        train_mask = raw_frame["season"] < year
        valid_mask = raw_frame["season"] == year
        prior = float(y.loc[train_mask].mean())

        context_features = select_v2_features(add_row_features(hierarchical, prior))
        context_features = add_trackman_features(context_features, trackman)
        context_features = add_context_trackman_features(context_features, context_trackman)
        base_ctx = [c for c in context_features.columns if c not in MATCHUP_HTE]
        with_ins = pd.concat([context_features, inseason_block], axis=1)
        ins_ctx = [c for c in with_ins.columns if c not in MATCHUP_HTE]
        variants = {
            "inseason": (with_ins, ins_ctx),
            "noseason": (context_features, [c for c in base_ctx if c != "season"]),
            "inseason_noseason": (with_ins, [c for c in ins_ctx if c != "season"]),
        }
        for name, (frame, columns) in variants.items():
            ctx[name][str(year)] = fit_predict(
                frame, columns, None, train_mask, valid_mask, y
            )
            counts[f"context_{name}"] = len(columns)
            print(f"year={year} context={name} ({len(columns)} feat)", flush=True)

        form_features = select_v2_features(add_row_features(form_raw, prior))
        form_features = add_trackman_features(form_features, trackman)
        form_features = pd.concat([form_features, inseason_block], axis=1)
        form_cols = [c for c in v31_form_columns(form_features) if c != "season"]
        form_noseason[str(year)] = fit_predict(
            form_features, form_cols, FORM_CONFIG, train_mask, valid_mask, y
        )
        counts["form_noseason"] = len(form_cols)
        print(f"year={year} form=noseason ({len(form_cols)} feat)", flush=True)

    validation_frame = make_validation_frame()
    unit_correction = np.concatenate([
        drift_correction(inseason_full.loc[oof[str(year)]["row_index"]], 1.0)
        for year in YEARS
    ])

    def assemble(form, context):
        parts = []
        for year in YEARS:
            prediction, _, _, _ = calibrated_prediction(
                year, oof, logistic, form, context, raw_frame
            )
            parts.append(prediction)
        return np.clip(np.concatenate(parts) + DRIFT_WEIGHT * unit_correction, 0, 1)

    v93_baseline = assemble(v93_form, v93_context)
    reference = validation_frame.copy()
    reference["v41_prediction"] = v93_baseline
    reference["v41_squared_error"] = (v93_baseline - reference["target"]) ** 2

    combos = {
        "H1_ctx_inseason": (v93_form, ctx["inseason"]),
        "H2_ctx_noseason": (v93_form, ctx["noseason"]),
        "H2_form_noseason": (form_noseason, v93_context),
        "H2_both_noseason": (form_noseason, ctx["noseason"]),
        "H1H2_ctx_inseason_noseason": (v93_form, ctx["inseason_noseason"]),
        "H1H2_form_noseason_ctx_inseason_noseason": (
            form_noseason, ctx["inseason_noseason"]
        ),
    }
    candidates, development = {}, {}
    for label, (form, context) in combos.items():
        candidates[label] = assemble(form, context)
        development[label] = development_metrics(reference, candidates[label])

    eligible = [
        label for label, r in development.items()
        if r["season_gain_development"]["2022"] > -1e-5
        and r["season_gain_development"]["2023"] > -1e-5
        and r["gain_2024_mar_aug"] >= MIN_GAIN
        and r["gain_2024_jul_aug"] > 0
        and r["monthly_block_win_rate"] >= 0.75
    ]
    promoted = max(
        eligible, key=lambda label: development[label]["gain_2024_mar_aug"], default=None
    )
    final_confirmation = None
    if promoted is not None:
        final_confirmation = {
            "candidate": promoted,
            "comparison": compare_candidate(reference, candidates[promoted]),
        }

    output = {
        "experiment": "V94a_context_inseason_and_season_removal",
        "baseline": "V93",
        "baseline_public_score": 951.1067955895,
        "hypotheses": {
            "H1": "Context HGB (13% of the blend) has never seen the in-season reconstruction",
            "H2": "`season` is a raw numeric feature; trees cannot extrapolate to 2025",
        },
        "feature_counts": counts,
        "drift_weight": DRIFT_WEIGHT,
        "min_gain_for_submission": MIN_GAIN,
        "local_to_leaderboard_ratio": RATIO,
        "development_results": development,
        "eligible_candidates": eligible,
        "promoted_candidate": promoted,
        "final_confirmation_inspected": promoted is not None,
        "final_confirmation": final_confirmation,
        "compliance": {
            "official_data_only": True,
            "test_csv_read": False,
            "test_row_aggregation_used": False,
            "current_pitch_post_event_information_used": False,
            "trackman_2025_used": False,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"context": ctx, "form_noseason": form_noseason,
                 "v93_baseline": v93_baseline}, PREDICTIONS, compress=3)

    print(f"\nfeature counts: {counts}")
    print("gains vs V93 baseline, development windows only:")
    print(f"{'candidate':>42} {'2022':>8} {'2023':>8} {'2024pt':>8} {'2024 gain':>12} {'LB est':>7} {'blocks':>8} {'worst':>12}")
    for label, r in sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"]):
        sg = r["season_gain_development"]
        print(f"{label:>42} {points(sg['2022']):8.1f} {points(sg['2023']):8.1f} "
              f"{points(r['gain_2024_mar_aug']):8.1f} {r['gain_2024_mar_aug']:12.3e} "
              f"{points(r['gain_2024_mar_aug'])*RATIO:7.1f} "
              f"{r['monthly_block_win_rate']:8.2%} {r['worst_monthly_gain']:12.3e}")
    print(f"\neligible={eligible}\npromoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
