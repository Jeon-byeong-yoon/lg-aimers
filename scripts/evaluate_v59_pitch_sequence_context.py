"""V59: Incorporate Pitch of PA & Pitch Count fatigue buckets into Context HGB.

Hypothesis:
-----------
In baseball, pitch location & control quality are strongly influenced by:
1. `pitch_of_pa` (1st pitch attack vs 2-strike finishing pitch)
2. `pitch_no` bucket (1~25 early, 26~50 middle, 51~75 fatigue onset, 76+ deep fatigue)

`trackman_history.csv`, `train.csv`, and `test.csv` all contain both `pitch_of_pa` and `pitch_no`.
Adding `pitch_of_pa` or `pitch_count_bucket` into the Context aggregation keys
allows Context HGB to capture pitch sequence dynamics and pitcher fatigue trends.

Evaluation:
  Strict 3-season chronological validation (2022 raw OOF, 2023 calib, 2024 calib) with V41 weights (0.55 / 0.32 / 0.13).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v2 import hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import KEYS, PHYSICAL, PITCH_GROUPS, add_trackman_features, prepare_trackman


# V41 outer blend weights
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13

V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

VALUE_COLUMNS = PHYSICAL + [f"pitch_group_{group}" for group in PITCH_GROUPS]
SMOOTHING = (100.0, 500.0)


def add_sequence_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["inning_bucket"] = pd.cut(
        output["inning"], bins=[-1, 3, 6, 9, np.inf],
        labels=False, include_lowest=True,
    ).astype("int8")
    # Pitch fatigue bucket: 0: 1-30, 1: 31-60, 2: 61-90, 3: 91+
    output["pitch_fatigue_bucket"] = pd.cut(
        output["pitch_no"], bins=[-1, 30, 60, 90, np.inf],
        labels=False, include_lowest=True,
    ).astype("int8")
    # Pitch of PA cap at 6
    output["pitch_pa_capped"] = np.clip(output["pitch_of_pa"], 1, 6).astype("int8")
    return output


def build_custom_context_lookup(trackman: pd.DataFrame, before_season: int, context_cols: list) -> pd.DataFrame:
    history = trackman.loc[
        (trackman["season"] < before_season)
        & trackman["outs_before"].between(0, 2)
        & trackman["top_bottom"].isin(["T", "B"])
    ]
    detail_keys = KEYS + context_cols
    parent = history.groupby(KEYS, observed=True)[VALUE_COLUMNS].mean().reset_index()
    parent = parent.rename(columns={column: f"__parent_{column}" for column in VALUE_COLUMNS})
    detail = history.groupby(detail_keys, observed=True)[VALUE_COLUMNS].agg(["mean", "count"])
    detail.columns = [f"{column}__{stat}" for column, stat in detail.columns]
    detail = detail.reset_index().merge(parent, how="left", on=KEYS, validate="many_to_one")
    output = detail[detail_keys].copy()
    counts = detail[f"{VALUE_COLUMNS[0]}__count"].astype(float)
    output["tm_ctx_history_n"] = counts
    output["tm_ctx_log1p_n"] = np.log1p(counts)
    for column in VALUE_COLUMNS:
        mean = detail[f"{column}__mean"]
        parent_mean = detail[f"__parent_{column}"]
        for smoothing in SMOOTHING:
            output[f"tm_ctx_{column}_s{int(smoothing)}"] = (
                mean * counts + parent_mean * smoothing
            ) / (counts + smoothing)
    return output


def add_custom_context_features(frame: pd.DataFrame, trackman: pd.DataFrame, context_cols: list) -> pd.DataFrame:
    source = add_sequence_keys(frame)
    pieces = []
    for season in sorted(source["season"].unique()):
        rows = source.loc[source["season"] == season].copy()
        rows["__original_index"] = rows.index
        lookup = build_custom_context_lookup(trackman, season, context_cols)
        if lookup.empty:
            for column in VALUE_COLUMNS:
                for smoothing in SMOOTHING:
                    rows[f"tm_ctx_{column}_s{int(smoothing)}"] = float("nan")
            rows["tm_ctx_history_n"] = float("nan")
            rows["tm_ctx_log1p_n"] = float("nan")
        else:
            rows = rows.merge(lookup, how="left", on=KEYS + context_cols, validate="many_to_one")
        pieces.append(rows.set_index("__original_index"))
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    return output


def v31_context_columns(frame):
    return [c for c in frame.columns if not c.startswith("hte_pitcher_batter_")]


def main():
    print("Loading data for V59...", flush=True)
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id").copy()
    
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    
    raw_trackman = pd.read_csv("공모전 dataset/open/data/trackman_history.csv")
    trackman_base = prepare_trackman(raw_trackman)
    trackman_seq = add_sequence_keys(trackman_base)
    trackman_seq["top_bottom"] = trackman_seq["top_bottom"].map({"Top": "T", "Bottom": "B"})
    
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form_preds = joblib.load("artifacts/v38_lr_grid_predictions.joblib")["gentle_500"]
    
    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in (2022, 2023, 2024)}
    frames = {str(yr): raw_frame.loc[oof[str(yr)]["row_index"]] for yr in (2022, 2023, 2024)}
    v11_tree_preds = {str(yr): v11_prediction(oof[str(yr)]) for yr in (2022, 2023, 2024)}
    v17_preds = {str(yr): 0.95 * v11_tree_preds[str(yr)] + 0.05 * logistic[str(yr)] for yr in (2022, 2023, 2024)}
    
    f23_train_frame = frames["2022"]
    f24_train_frame = pd.concat([frames["2022"], frames["2023"]])
    
    # Experiments:
    # 1. ctx_plus_pitch_pa: ["outs_before", "inning_bucket", "top_bottom", "pitch_pa_capped"]
    # 2. ctx_plus_fatigue: ["outs_before", "inning_bucket", "top_bottom", "pitch_fatigue_bucket"]
    # 3. ctx_pa_only: ["outs_before", "pitch_pa_capped"]
    variants = {
        "ctx_plus_pitch_pa": ["outs_before", "inning_bucket", "top_bottom", "pitch_pa_capped"],
        "ctx_plus_fatigue": ["outs_before", "inning_bucket", "top_bottom", "pitch_fatigue_bucket"],
        "ctx_plus_pa_and_fatigue": ["outs_before", "inning_bucket", "top_bottom", "pitch_pa_capped", "pitch_fatigue_bucket"],
    }
    
    results = []
    
    for exp_name, ctx_keys in variants.items():
        print(f"Evaluating Context variant: {exp_name}...", flush=True)
        ctx_feat = add_custom_context_features(hierarchical, trackman_seq, ctx_keys)
        ctx_feat = add_trackman_features(ctx_feat, trackman_base)
        
        ctx_preds = {}
        for year in (2022, 2023, 2024):
            train_mask = data["season"] < year
            valid_mask = data["season"] == year
            prior = float(y.loc[train_mask].mean())
            
            feat_year = select_v2_features(add_row_features(ctx_feat, prior))
            cols = v31_context_columns(feat_year)
            candidate = feat_year[cols]
            
            model, feat_cols = hist_gbdt_pipeline(candidate)
            model.fit(candidate.loc[train_mask, feat_cols], y.loc[train_mask])
            ctx_preds[str(year)] = model.predict_proba(candidate.loc[valid_mask, feat_cols])[:, 1]
            
        raw_blend = {
            str(yr): W_V17 * v17_preds[str(yr)] + W_FORM * form_preds[str(yr)] + W_CONTEXT * ctx_preds[str(yr)]
            for yr in (2022, 2023, 2024)
        }
        
        score22 = float(brier_score_loss(targets["2022"], raw_blend["2022"]))
        
        # 2023 calib
        res22 = targets["2022"] - raw_blend["2022"]
        cnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["balls_before", "strikes_before"], 500)
        pcnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred23_cal = np.clip(raw_blend["2023"] + res22.mean() + 0.75 * cnt_23 + 0.25 * pcnt_23, 0, 1)
        score23 = float(brier_score_loss(targets["2023"], pred23_cal))
        
        # 2024 calib
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
            "context_keys": ctx_keys,
            "scores": {"2022": score22, "2023": score23, "2024": score24},
            "gains_vs_v41": {"2022": gain22, "2023": gain23, "2024": gain24},
            "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
            "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
        })
        print(f"  -> {exp_name}: gains 2022={gain22:+.8f}, 2023={gain23:+.8f}, 2024={gain24:+.8f}", flush=True)
        
    accepted = [r for r in results if r["all_improved"]]
    submit_ready = [r for r in results if r["submit_ready"]]
    
    output = {
        "experiment": "V59_pitch_sequence_context",
        "description": "Context HGB with pitch_of_pa and pitch_fatigue_bucket context keys",
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "candidate_count": len(results),
        "accepted_count": len(accepted),
        "submit_ready_count": len(submit_ready),
        "best_accepted": accepted[0] if accepted else None,
        "best_submit_ready": submit_ready[0] if submit_ready else None,
        "all_results": results,
    }
    
    Path("artifacts/v59_pitch_sequence_context_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
