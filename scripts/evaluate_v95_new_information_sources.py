"""V95: two feature groups that existing columns cannot reproduce.

V94 established the filter this step is built on. Recentering target encodings
and dropping `season` both failed because they only *re-expressed* information
the model already had. V92/V93 succeeded because the current-season
reconstruction was genuinely new. So every candidate here must answer yes to:
"is this information absent from the current feature set?"

A — Multi-season trajectory. The model sees the career-cumulative rate and the
current-season rate, but nothing says whether a pitcher has declined for three
straight seasons or is stable. Differencing adjacent frozen season-end anchors
recovers each individual season's rate, hence a trajectory.

B — Current-season pitch mix. `asof_pitcher_pitchmix_n` and the fastball /
breaking / offspeed rates are also career-cumulative and were deliberately left
out of V93. A pitcher whose repertoire changed this season (injury, a move to
relief) is invisible in the career mix.

Baseline is V93 (Public 951.1067955895). The Context in-season variant held over
from V94a is re-tested in combination, since its information source differs.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
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
from season_trajectory_v95 import FEATURE_NAMES as TRAJ, build_training_trajectory
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUTPUT = Path("artifacts/v95_new_information_metrics.json")
PREDICTIONS = Path("artifacts/v95_new_information_predictions.joblib")
FORM_CONFIG = {**BASE_CONFIG, "learning_rate": 0.03, "max_iter": 500}
DRIFT_WEIGHT = 0.20
BASE_INSEASON = feature_names(("pitcher", "batter"))
PITCHMIX = [c for c in feature_names(("pitcher", "batter", "pitchmix"))
            if c not in BASE_INSEASON]
MIN_GAIN = 1.06e-5
RATIO = 0.944


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

    inseason_full = add_training_inseason_features(
        raw_frame, groups=("pitcher", "batter", "pitchmix")
    )
    inseason_block = inseason_full[BASE_INSEASON]
    pitchmix_block = inseason_full[PITCHMIX]
    traj_block = build_training_trajectory(raw_frame)[TRAJ]
    print(f"new feature groups: trajectory={len(TRAJ)}, pitchmix={len(PITCHMIX)}", flush=True)

    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    v93_form = joblib.load("artifacts/v93_inseason_form_model_predictions.joblib")
    v93_form = v93_form["new_form"]["with_2019_nan"]
    v93_context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    v93_context = v93_context["no_matchup_hte"]["context"]
    v94a = joblib.load("artifacts/v94a_context_inseason_predictions.joblib")
    ctx_inseason = v94a["context"]["inseason"]

    extras = {
        "traj": [traj_block],
        "pitchmix": [pitchmix_block],
        "traj_pitchmix": [traj_block, pitchmix_block],
    }
    forms = {name: {} for name in extras}
    counts = {}
    for year in YEARS:
        train_mask = raw_frame["season"] < year
        valid_mask = raw_frame["season"] == year
        prior = float(y.loc[train_mask].mean())
        base = select_v2_features(add_row_features(form_raw, prior))
        base = add_trackman_features(base, trackman)
        base = pd.concat([base, inseason_block], axis=1)
        for name, blocks in extras.items():
            frame = pd.concat([base, *blocks], axis=1)
            columns = v31_form_columns(frame)
            counts[name] = len(columns)
            model, model_columns = hist_gbdt_pipeline(frame[columns])
            model.set_params(**{
                f"histgradientboostingclassifier__{k}": v for k, v in FORM_CONFIG.items()
            })
            model.fit(frame.loc[train_mask, model_columns], y.loc[train_mask])
            forms[name][str(year)] = model.predict_proba(
                frame.loc[valid_mask, model_columns])[:, 1]
            print(f"year={year} form={name} ({len(columns)} feat)", flush=True)

    validation_frame = make_validation_frame()
    unit_correction = np.concatenate([
        drift_correction(inseason_full.loc[oof[str(year)]["row_index"]], 1.0)
        for year in YEARS
    ])

    def assemble(form, context):
        pieces = []
        for year in YEARS:
            prediction, _, _, _ = calibrated_prediction(
                year, oof, logistic, form, context, raw_frame)
            pieces.append(prediction)
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * unit_correction, 0, 1)

    v93_baseline = assemble(v93_form, v93_context)
    reference = validation_frame.copy()
    reference["v41_prediction"] = v93_baseline
    reference["v41_squared_error"] = (v93_baseline - reference["target"]) ** 2

    candidates, development = {}, {}
    for form_name, form in list(forms.items()) + [("v93_form", v93_form)]:
        for ctx_name, context in (("v93_ctx", v93_context), ("ctx_inseason", ctx_inseason)):
            if form_name == "v93_form" and ctx_name == "v93_ctx":
                continue
            label = f"{form_name}__{ctx_name}"
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
    promoted = max(eligible, key=lambda l: development[l]["gain_2024_mar_aug"], default=None)
    final_confirmation = None
    if promoted is not None:
        final_confirmation = {"candidate": promoted,
                              "comparison": compare_candidate(reference, candidates[promoted])}

    output = {
        "experiment": "V95_new_information_sources",
        "baseline": "V93",
        "baseline_public_score": 951.1067955895,
        "new_information_filter": (
            "candidate must carry information the current feature set cannot reproduce; "
            "V94 showed re-expressions of existing information do not transfer"
        ),
        "trajectory_features": TRAJ,
        "pitchmix_features": PITCHMIX,
        "form_feature_counts": counts,
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
            "anchors_frozen_at_training_time": True,
            "row_independent_transform": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"forms": forms, "v93_baseline": v93_baseline}, PREDICTIONS, compress=3)

    print("\ngains vs V93 baseline, development windows only:")
    print(f"{'candidate':>34} {'2022':>8} {'2023':>8} {'2024pt':>8} {'2024 gain':>12} {'LB est':>7} {'blocks':>8} {'jul_aug':>12} {'worst':>12}")
    for label, r in sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"]):
        sg = r["season_gain_development"]
        print(f"{label:>34} {points(sg['2022']):8.1f} {points(sg['2023']):8.1f} "
              f"{points(r['gain_2024_mar_aug']):8.1f} {r['gain_2024_mar_aug']:12.3e} "
              f"{points(r['gain_2024_mar_aug'])*RATIO:7.1f} "
              f"{r['monthly_block_win_rate']:8.2%} {r['gain_2024_jul_aug']:12.3e} "
              f"{r['worst_monthly_gain']:12.3e}")
    print(f"\neligible={eligible}\npromoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
