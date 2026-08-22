"""Evaluate advanced physical Stuff metrics in Context HGB on top of V41.

Hypothesis & Mathematics:
-------------------------
In Trackman analytics:
1. Bauer Units = spin_rate / rel_speed
   - Measures spin efficiency per velocity unit (crucial for predicting vertical rise / control).
2. Total Movement = sqrt(induced_vert_break^2 + horz_break^2)
   - Measures the total deviation of the pitch from a ballistic trajectory.
3. Velocity Drop = rel_speed - zone_speed
   - Measures air resistance / drag and deceleration toward home plate.
4. Release Radius = sqrt(rel_height^2 + rel_side^2)
   - Measures spatial release point consistency.

Adding these 4 derived physical metrics into Contextual Trackman allows Context HGB (13% weight)
to capture non-linear physical pitching characteristics with strict prior-season integrity.

Evaluation:
  Full V41 3-way blend (0.55 / 0.32 / 0.13) + expanding calibration across
  2022 raw OOF, 2023 expanding calib, and 2024 expanding calib.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v2 import hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import KEYS, PITCH_GROUPS, add_trackman_features, prepare_trackman


# V41 outer blend weights
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13

# V41 baseline scores
V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

CONTEXT_KEYS = ["outs_before", "inning_bucket", "top_bottom"]
BASE_PHYSICAL = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
    "rel_height", "rel_side", "zone_speed",
]
STUFF_PHYSICAL = [
    "bauer_units", "total_break", "speed_drop", "release_radius"
]
SMOOTHING = (100.0, 500.0)


def prepare_stuff_trackman(trackman: pd.DataFrame) -> pd.DataFrame:
    data = prepare_trackman(trackman)
    data["inning_bucket"] = pd.cut(
        data["inning"], bins=[-1, 3, 6, 9, np.inf],
        labels=False, include_lowest=True,
    ).astype("int8")
    data["top_bottom"] = data["top_bottom"].map({"Top": "T", "Bottom": "B"})
    
    # Derived physical metrics
    data["bauer_units"] = data["spin_rate"] / np.clip(data["rel_speed"], 50.0, 120.0)
    data["total_break"] = np.sqrt(data["induced_vert_break"]**2 + data["horz_break"]**2)
    data["speed_drop"] = data["rel_speed"] - data["zone_speed"]
    data["release_radius"] = np.sqrt(data["rel_height"]**2 + data["rel_side"]**2)
    return data


def build_stuff_context_lookup(trackman: pd.DataFrame, before_season: int, value_cols: list) -> pd.DataFrame:
    history = trackman.loc[
        (trackman["season"] < before_season)
        & trackman["outs_before"].between(0, 2)
        & trackman["top_bottom"].isin(["T", "B"])
    ]
    detail_keys = KEYS + CONTEXT_KEYS
    parent = history.groupby(KEYS, observed=True)[value_cols].mean().reset_index()
    parent = parent.rename(columns={column: f"__parent_{column}" for column in value_cols})
    detail = history.groupby(detail_keys, observed=True)[value_cols].agg(["mean", "count"])
    detail.columns = [f"{column}__{stat}" for column, stat in detail.columns]
    detail = detail.reset_index().merge(parent, how="left", on=KEYS, validate="many_to_one")
    output = detail[detail_keys].copy()
    counts = detail[f"{value_cols[0]}__count"].astype(float)
    output["tm_ctx_history_n"] = counts
    output["tm_ctx_log1p_n"] = np.log1p(counts)
    for column in value_cols:
        mean = detail[f"{column}__mean"]
        parent_mean = detail[f"__parent_{column}"]
        for smoothing in SMOOTHING:
            output[f"tm_ctx_{column}_s{int(smoothing)}"] = (
                mean * counts + parent_mean * smoothing
            ) / (counts + smoothing)
    return output


def add_stuff_context_features(frame: pd.DataFrame, trackman: pd.DataFrame, value_cols: list) -> pd.DataFrame:
    source = frame.copy()
    source["inning_bucket"] = pd.cut(
        source["inning"], bins=[-1, 3, 6, 9, np.inf],
        labels=False, include_lowest=True,
    ).astype("int8")
    pieces = []
    for season in sorted(source["season"].unique()):
        rows = source.loc[source["season"] == season].copy()
        rows["__original_index"] = rows.index
        lookup = build_stuff_context_lookup(trackman, season, value_cols)
        if lookup.empty:
            for column in value_cols:
                for smoothing in SMOOTHING:
                    rows[f"tm_ctx_{column}_s{int(smoothing)}"] = float("nan")
            rows["tm_ctx_history_n"] = float("nan")
            rows["tm_ctx_log1p_n"] = float("nan")
        else:
            rows = rows.merge(lookup, how="left", on=KEYS + CONTEXT_KEYS, validate="many_to_one")
        pieces.append(rows.set_index("__original_index"))
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    return output


def v31_context_columns(frame):
    return [c for c in frame.columns if not c.startswith("hte_pitcher_batter_")]


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    
    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv",
        usecols=BASE_PHYSICAL + ["season", "pitcher_hand", "batter_hand", "game_month", "balls_before", "strikes_before", "outs_before", "inning", "top_bottom", "pitch_type_group"]
    )
    stuff_trackman = prepare_stuff_trackman(raw_trackman)
    trackman_base = prepare_trackman(raw_trackman)
    
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    
    v17_preds = {
        str(yr): 0.95 * v11_prediction(oof[str(yr)]) + 0.05 * logistic[str(yr)]
        for yr in (2022, 2023, 2024)
    }
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in (2022, 2023, 2024)}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in (2022, 2023, 2024)}
    
    # Feature sets to test:
    # 1. base_context: BASE_PHYSICAL + PITCH_GROUPS (V41 baseline)
    # 2. plus_stuff: BASE_PHYSICAL + STUFF_PHYSICAL + PITCH_GROUPS
    # 3. stuff_only: STUFF_PHYSICAL + PITCH_GROUPS
    experiments = {
        "plus_stuff_metrics": BASE_PHYSICAL + STUFF_PHYSICAL + [f"pitch_group_{g}" for g in PITCH_GROUPS],
        "stuff_core_metrics": BASE_PHYSICAL + ["bauer_units", "total_break"] + [f"pitch_group_{g}" for g in PITCH_GROUPS],
    }
    
    results = []
    f23_train_frame = frames["2022"]
    f24_train_frame = pd.concat([frames["2022"], frames["2023"]])
    
    for exp_name, val_cols in experiments.items():
        print(f"Training Context HGB variant: {exp_name}...", flush=True)
        # Build features with new context metrics
        context_features = add_stuff_context_features(hierarchical, stuff_trackman, val_cols)
        context_features = add_trackman_features(context_features, trackman_base)
        
        context_preds_exp = {}
        for year in (2022, 2023, 2024):
            train_mask = data["season"] < year
            valid_mask = data["season"] == year
            prior = float(y.loc[train_mask].mean())
            
            feat_year = select_v2_features(add_row_features(context_features, prior))
            cols = v31_context_columns(feat_year)
            candidate = feat_year[cols]
            
            model, feat_cols = hist_gbdt_pipeline(candidate)
            model.fit(candidate.loc[train_mask, feat_cols], y.loc[train_mask])
            context_preds_exp[str(year)] = model.predict_proba(candidate.loc[valid_mask, feat_cols])[:, 1]
            
        raw_blend = {
            str(yr): W_V17 * v17_preds[str(yr)] + W_FORM * form_preds[str(yr)] + W_CONTEXT * context_preds_exp[str(yr)]
            for yr in (2022, 2023, 2024)
        }
        score22 = float(brier_score_loss(targets["2022"], raw_blend["2022"]))
        
        # Fast Fold 2023
        res22 = targets["2022"] - raw_blend["2022"]
        cnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["balls_before", "strikes_before"], 500)
        pcnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred23_cal = np.clip(raw_blend["2023"] + res22.mean() + 0.75 * cnt_23 + 0.25 * pcnt_23, 0, 1)
        score23 = float(brier_score_loss(targets["2023"], pred23_cal))
        
        # Fast Fold 2024
        res_22_23 = np.concatenate([res22, targets["2023"] - raw_blend["2023"]])
        cnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["balls_before", "strikes_before"], 500)
        pcnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred24_cal = np.clip(raw_blend["2024"] + res_22_23.mean() + 0.75 * cnt_24 + 0.25 * pcnt_24, 0, 1)
        score24 = float(brier_score_loss(targets["2024"], pred24_cal))
        
        gain22 = V41_BASELINE[2022] - score22
        gain23 = V41_BASELINE[2023] - score23
        gain24 = V41_BASELINE[2024] - score24
        
        results.append({
            "variant": exp_name,
            "scores": {"2022": score22, "2023": score23, "2024": score24},
            "gains_vs_v41": {"2022": gain22, "2023": gain23, "2024": gain24},
            "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
            "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
        })
        print(f"variant={exp_name} gains: 2022={gain22:.8f}, 2023={gain23:.8f}, 2024={gain24:.8f}", flush=True)
        
    accepted = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]
    
    output = {
        "experiment": "V52_context_physical_stuff",
        "description": "Evaluate derived physical Stuff metrics (Bauer Units, Total Break, Speed Drop) in Context HGB",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": results,
    }
    
    Path("artifacts/v52_context_stuff_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
