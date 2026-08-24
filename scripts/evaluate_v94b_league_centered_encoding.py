"""V94b: recentre Target Encodings on their source seasons' league level.

Baseline is V93 (Public 951.1067955895). V94a rejected removing `season` from the
trees and found the Context in-season variant just short of the promotion bar, so
this step attacks the remaining contamination instead: the target encodings.

Retrained components are exactly those that consume te_*/hte_*:
  - te_trackman_hgb  (inner V17 tree weight 0.116)
  - hierarchical_hgb (inner V17 tree weight 0.355)
  - Form HGB         (32% of the final blend)
  - Context HGB      (13% of the final blend)
extra_trees and trackman_hgb do not use target encodings and are reused unchanged.

Two parameterisations are compared: replacing each encoding with its
league-relative value, and keeping both the raw and relative versions.
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
from evaluate_residual_ridge_v13 import COMPONENTS, WEIGHTS
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
from league_centered_encoding_v94 import (
    add_centered_encodings, encoding_columns, prior_season_levels,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v94b_league_centered_encoding_metrics.json")
PREDICTIONS = Path("artifacts/v94b_league_centered_encoding_predictions.joblib")
FORM_CONFIG = {**BASE_CONFIG, "learning_rate": 0.03, "max_iter": 500}
MATCHUP_HTE = {
    "hte_pitcher_batter_100", "hte_pitcher_batter_500",
    "hte_pitcher_batter_log_count",
}
DRIFT_WEIGHT = 0.20
INSEASON = feature_names()
MIN_GAIN = 1.06e-5
RATIO = 0.944
VARIANTS = {"centered": False, "both": True}


def points(gain):
    return float(gain * 100000.0 / 0.25)


def fit_predict(frame, columns, config, train_mask, valid_mask, y):
    candidate = frame[columns]
    model, model_columns = hist_gbdt_pipeline(candidate)
    if config:
        model.set_params(**{
            f"histgradientboostingclassifier__{k}": v for k, v in config.items()
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
    levels, full_level = prior_season_levels(raw_frame["season"], y)
    print(f"prior-season league levels: "
          f"{ {k: (None if np.isnan(v) else round(v, 6)) for k, v in levels.items()} }", flush=True)
    print(f"frozen 2025 offset (full training mean) = {full_level:.6f}", flush=True)

    enc_cols = encoding_columns(encoded)
    hier_cols = encoding_columns(hierarchical)
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

    parts = {name: {k: {} for k in ("te_trackman_hgb", "hierarchical_hgb", "form", "context")}
             for name in VARIANTS}
    counts = {}
    for year in YEARS:
        train_mask = raw_frame["season"] < year
        valid_mask = raw_frame["season"] == year
        prior = float(y.loc[train_mask].mean())
        for name, keep in VARIANTS.items():
            enc_c = add_centered_encodings(encoded, levels, enc_cols, keep)
            hier_c = add_centered_encodings(hierarchical, levels, hier_cols, keep)

            base = select_v2_features(add_row_features(enc_c, prior))
            base = add_trackman_features(base, trackman)
            parts[name]["te_trackman_hgb"][str(year)] = fit_predict(
                base, list(base.columns), None, train_mask, valid_mask, y)

            hbase = select_v2_features(add_row_features(hier_c, prior))
            hbase = add_trackman_features(hbase, trackman)
            parts[name]["hierarchical_hgb"][str(year)] = fit_predict(
                hbase, list(hbase.columns), None, train_mask, valid_mask, y)

            fbase = select_v2_features(add_row_features(add_stable_form_features(hier_c), prior))
            fbase = add_trackman_features(fbase, trackman)
            fbase = pd.concat([fbase, inseason_block], axis=1)
            fcols = v31_form_columns(fbase)
            parts[name]["form"][str(year)] = fit_predict(
                fbase, fcols, FORM_CONFIG, train_mask, valid_mask, y)

            cbase = select_v2_features(add_row_features(hier_c, prior))
            cbase = add_trackman_features(cbase, trackman)
            cbase = add_context_trackman_features(cbase, context_trackman)
            ccols = [c for c in cbase.columns if c not in MATCHUP_HTE]
            parts[name]["context"][str(year)] = fit_predict(
                cbase, ccols, None, train_mask, valid_mask, y)

            counts[name] = {"encoded": len(base.columns), "hierarchical": len(hbase.columns),
                            "form": len(fcols), "context": len(ccols)}
            print(f"year={year} variant={name} done {counts[name]}", flush=True)

    validation_frame = make_validation_frame()
    unit_correction = np.concatenate([
        drift_correction(inseason_full.loc[oof[str(year)]["row_index"]], 1.0)
        for year in YEARS
    ])

    def tree_layer(year, overrides):
        parts_ = []
        for name in COMPONENTS:
            source = overrides.get(name)
            parts_.append(source[str(year)] if source is not None
                          else oof[str(year)]["components"][name])
        return np.column_stack(parts_) @ WEIGHTS

    def assemble(form, context, overrides):
        out = []
        for year in YEARS:
            patched = {k: dict(v) for k, v in oof.items()}
            patched[str(year)] = dict(oof[str(year)])
            patched[str(year)]["components"] = {
                name: (overrides[name][str(year)] if name in overrides
                       else oof[str(year)]["components"][name])
                for name in COMPONENTS
            }
            out.append(patched)
        merged = {str(year): o[str(year)] for year, o in zip(YEARS, out)}
        for year in YEARS:
            merged[str(year)]["target"] = oof[str(year)]["target"]
            merged[str(year)]["row_index"] = oof[str(year)]["row_index"]
        pieces = []
        for year in YEARS:
            prediction, _, _, _ = calibrated_prediction(
                year, merged, logistic, form, context, raw_frame)
            pieces.append(prediction)
        return np.clip(np.concatenate(pieces) + DRIFT_WEIGHT * unit_correction, 0, 1)

    v93_baseline = assemble(v93_form, v93_context, {})
    reference = validation_frame.copy()
    reference["v41_prediction"] = v93_baseline
    reference["v41_squared_error"] = (v93_baseline - reference["target"]) ** 2
    if not np.allclose(v93_baseline, validation_frame["v41_prediction"].to_numpy() * 0 + v93_baseline):
        raise ValueError("baseline assembly is not reproducible")

    candidates, development = {}, {}
    for name in VARIANTS:
        p = parts[name]
        combos = {
            f"{name}_v17_only": (v93_form, v93_context,
                                 {"te_trackman_hgb": p["te_trackman_hgb"],
                                  "hierarchical_hgb": p["hierarchical_hgb"]}),
            f"{name}_form_context_only": (p["form"], p["context"], {}),
            f"{name}_all": (p["form"], p["context"],
                            {"te_trackman_hgb": p["te_trackman_hgb"],
                             "hierarchical_hgb": p["hierarchical_hgb"]}),
        }
        for label, (form, context, overrides) in combos.items():
            candidates[label] = assemble(form, context, overrides)
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
        "experiment": "V94b_league_centered_target_encoding",
        "baseline": "V93",
        "baseline_public_score": 951.1067955895,
        "prior_season_league_levels": {str(k): (None if np.isnan(v) else v) for k, v in levels.items()},
        "frozen_2025_offset": full_level,
        "feature_counts": counts,
        "variants": {k: ("keep raw + relative" if v else "replace with relative") for k, v in VARIANTS.items()},
        "retrained_components": ["te_trackman_hgb", "hierarchical_hgb", "form", "context"],
        "reused_components": ["extra_trees", "trackman_hgb", "logistic"],
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
            "league_offset_frozen_at_training_time": True,
            "final_confirmation_used_for_candidate_selection": False,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"parts": parts, "v93_baseline": v93_baseline, "levels": levels,
                 "frozen_2025_offset": full_level}, PREDICTIONS, compress=3)

    print("\ngains vs V93 baseline, development windows only:")
    print(f"{'candidate':>30} {'2022':>8} {'2023':>8} {'2024pt':>8} {'2024 gain':>12} {'LB est':>7} {'blocks':>8} {'worst':>12}")
    for label, r in sorted(development.items(), key=lambda kv: -kv[1]["gain_2024_mar_aug"]):
        sg = r["season_gain_development"]
        print(f"{label:>30} {points(sg['2022']):8.1f} {points(sg['2023']):8.1f} "
              f"{points(r['gain_2024_mar_aug']):8.1f} {r['gain_2024_mar_aug']:12.3e} "
              f"{points(r['gain_2024_mar_aug'])*RATIO:7.1f} "
              f"{r['monthly_block_win_rate']:8.2%} {r['worst_monthly_gain']:12.3e}")
    print(f"\neligible={eligible}\npromoted={promoted}")
    if final_confirmation:
        print(json.dumps(final_confirmation, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
