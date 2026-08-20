"""V37-B: Stronger context Trackman smoothing (200, 1000) instead of (100, 500).

Rationale
---------
contextual_trackman_v24.py uses SMOOTHING = (100.0, 500.0) which has never been
tuned. Increasing to (200, 1000) shrinks the context features more aggressively
toward the parent (pitcher-hand × batter-hand × month × count) average, reducing
variance from small context samples (e.g. a pitcher seen only once in a specific
inning / out-state bucket). This is the same intuition that drove the form model
regularization in V35, applied independently to the context signal.

Experiment scope
----------------
- form    : V35-strong (same as V37a)
- context : rebuilt with SMOOTHING = (200, 1000)
- weights : unchanged (0.75 / 0.16 / 0.09)
- calibration : standard fold-based segment correction (same as fold_score_v37)

We also test the combination of both axes:
  - form=V35-strong, context=smoothing(200,1000)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from contextual_trackman_v24 import add_context_keys, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from evaluate_segment_calibration_v12 import segment_correction
from evaluate_v2 import hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import (
    KEYS, PHYSICAL, PITCH_GROUPS, add_trackman_features, prepare_trackman,
)


# V35-strong configuration (same as V37a)
FORM_CONFIG = {
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 20.0,
    "learning_rate": 0.05,
    "max_iter": 240,
}

MATCHUP_HTE = {
    "hte_pitcher_batter_100", "hte_pitcher_batter_500",
    "hte_pitcher_batter_log_count",
}

CONTEXT_DIMS = ["outs_before", "inning_bucket", "top_bottom"]
VALUE_COLUMNS = PHYSICAL + [f"pitch_group_{g}" for g in PITCH_GROUPS]

# Candidate smoothing grids for the context Trackman features
SMOOTHING_CANDIDATES = {
    "s100_500": (100.0, 500.0),   # V31 original
    "s200_1000": (200.0, 1000.0), # new: stronger shrinkage
    "s150_750": (150.0, 750.0),   # intermediate
}

V31_BASELINE = {
    2022: 0.24339835671689145,
    2023: 0.25326829650157456,
    2024: 0.24783690211517923,
}


def v17_prediction(item, logistic):
    return 0.95 * v11_prediction(item) + 0.05 * logistic


def v31_form_columns(frame):
    return [c for c in frame if not (c.startswith("tm_") and c.endswith("_std"))]


def v31_context_columns(frame):
    return [c for c in frame if c not in MATCHUP_HTE]


def build_context_lookup_custom(trackman: pd.DataFrame, before_season: int,
                                smoothing: tuple) -> pd.DataFrame:
    """Like contextual_trackman_v24.build_context_lookup but with custom smoothing."""
    history = trackman.loc[
        (trackman["season"] < before_season)
        & trackman["outs_before"].between(0, 2)
        & trackman["top_bottom"].isin(["T", "B"])
    ]
    detail_keys = KEYS + CONTEXT_DIMS
    parent = history.groupby(KEYS, observed=True)[VALUE_COLUMNS].mean().reset_index()
    parent = parent.rename(columns={c: f"__parent_{c}" for c in VALUE_COLUMNS})
    detail = history.groupby(detail_keys, observed=True)[VALUE_COLUMNS].agg(["mean", "count"])
    detail.columns = [f"{col}__{stat}" for col, stat in detail.columns.to_flat_index()]
    detail = detail.reset_index().merge(parent, how="left", on=KEYS, validate="many_to_one")
    output = detail[detail_keys].copy()
    counts = detail[f"{VALUE_COLUMNS[0]}__count"].astype(float)
    output["tm_ctx_history_n"] = counts
    output["tm_ctx_log1p_n"] = np.log1p(counts)
    for col in VALUE_COLUMNS:
        mean = detail[f"{col}__mean"]
        parent_mean = detail[f"__parent_{col}"]
        for s in smoothing:
            output[f"tm_ctx_{col}_s{int(s)}"] = (
                mean * counts + parent_mean * s
            ) / (counts + s)
    return output


def add_context_trackman_features_custom(frame: pd.DataFrame, trackman: pd.DataFrame,
                                         smoothing: tuple) -> pd.DataFrame:
    """Like contextual_trackman_v24.add_context_trackman_features but parameterised."""
    source = add_context_keys(frame)
    source["top_bottom"] = source["top_bottom"].map({"Top": "T", "Bottom": "B"})
    pieces = []
    for season in sorted(source["season"].unique()):
        rows = source.loc[source["season"] == season].copy()
        rows["__original_index"] = rows.index
        lookup = build_context_lookup_custom(trackman, int(season), smoothing)
        rows = rows.merge(lookup, how="left", on=KEYS + CONTEXT_DIMS, validate="many_to_one")
        pieces.append(rows.set_index("__original_index"))
    output = pd.concat(pieces).sort_index()
    output.index.name = frame.index.name
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise ValueError("Context Trackman join (custom smoothing) changed row identity")
    return output


def fold_score_v37(oof, logistic, form_preds, context_preds, raw_frame,
                   history_years, valid_year):
    indices, targets, predictions = [], [], []
    for year in history_years:
        item = oof[str(year)]
        v17 = v17_prediction(item, logistic[str(year)])
        blend = (
            0.75 * v17
            + 0.16 * form_preds[str(year)]
            + 0.09 * context_preds[str(year)]
        )
        predictions.append(blend)
        targets.append(item["target"].astype(float))
        indices.append(item["row_index"])
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    train_frame = raw_frame.loc[index]
    valid_item = oof[str(valid_year)]
    v17_valid = v17_prediction(valid_item, logistic[str(valid_year)])
    valid_pred_raw = (
        0.75 * v17_valid
        + 0.16 * form_preds[str(valid_year)]
        + 0.09 * context_preds[str(valid_year)]
    )
    valid_frame = raw_frame.loc[valid_item["row_index"]]
    valid_y = valid_item["target"].astype(float)
    count_corr = segment_correction(
        train_frame, residual, valid_frame,
        ["balls_before", "strikes_before"], 500,
    )
    pitcher_count_corr = segment_correction(
        train_frame, residual, valid_frame,
        ["pitcher_id", "balls_before", "strikes_before"], 300,
    )
    prediction = np.clip(
        valid_pred_raw + residual.mean() + 0.75 * count_corr + 0.25 * pitcher_count_corr,
        0, 1,
    )
    return float(brier_score_loss(valid_y, prediction))


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    )
    trackman = prepare_trackman(raw_trackman)
    # Prepare base trackman (with top_bottom mapping for context dims)
    context_trackman = prepare_context_trackman(trackman)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]

    # --- Form predictions (V35-strong, reused from V37a if already computed) ---
    # We recompute here for self-containment; same deterministic result.
    form_preds = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())
        form_raw = add_stable_form_features(hierarchical)
        form_features = select_v2_features(add_row_features(form_raw, prior))
        form_features = add_trackman_features(form_features, trackman)
        columns = v31_form_columns(form_features)
        candidate = form_features[columns]
        model, cols = hist_gbdt_pipeline(candidate)
        model.set_params(**{
            f"histgradientboostingclassifier__{k}": v for k, v in FORM_CONFIG.items()
        })
        model.fit(candidate.loc[train_mask, cols], y.loc[train_mask])
        form_preds[str(year)] = model.predict_proba(candidate.loc[valid_mask, cols])[:, 1]
        print(f"form year={year} done", flush=True)

    # --- Context predictions for each smoothing variant ---
    all_context_preds = {}
    for sname, smoothing in SMOOTHING_CANDIDATES.items():
        ctx_preds = {}
        for year in (2022, 2023, 2024):
            train_mask = data["season"] < year
            valid_mask = data["season"] == year
            prior = float(y.loc[train_mask].mean())
            ctx_features = select_v2_features(add_row_features(hierarchical, prior))
            ctx_features = add_trackman_features(ctx_features, trackman)
            ctx_features = add_context_trackman_features_custom(
                ctx_features, context_trackman, smoothing
            )
            columns = v31_context_columns(ctx_features)
            candidate = ctx_features[columns]
            model, cols = hist_gbdt_pipeline(candidate)
            model.fit(candidate.loc[train_mask, cols], y.loc[train_mask])
            ctx_preds[str(year)] = model.predict_proba(candidate.loc[valid_mask, cols])[:, 1]
            print(f"context smoothing={sname} year={year} done", flush=True)
        all_context_preds[sname] = ctx_preds

    # --- Evaluate ---
    results = []
    for sname, context_preds in all_context_preds.items():
        # 2022 raw OOF
        item22 = oof["2022"]
        v17_22 = v17_prediction(item22, logistic["2022"])
        pred22 = (
            0.75 * v17_22
            + 0.16 * form_preds["2022"]
            + 0.09 * context_preds["2022"]
        )
        score22 = float(brier_score_loss(item22["target"].astype(float), pred22))
        score23 = fold_score_v37(
            oof, logistic, form_preds, context_preds, raw_frame,
            history_years=[2022], valid_year=2023,
        )
        score24 = fold_score_v37(
            oof, logistic, form_preds, context_preds, raw_frame,
            history_years=[2022, 2023], valid_year=2024,
        )
        scores = {2022: score22, 2023: score23, 2024: score24}
        gains = {y: V31_BASELINE[y] - scores[y] for y in (2022, 2023, 2024)}
        accepted = all(gains[y] > 0 for y in (2022, 2023, 2024))
        submit_ready = accepted and gains[2024] > 5e-6
        results.append({
            "smoothing": sname,
            "smoothing_values": list(SMOOTHING_CANDIDATES[sname]),
            "scores": {str(k): v for k, v in scores.items()},
            "gains_vs_v31": {str(k): v for k, v in gains.items()},
            "all_seasons_improved": accepted,
            "submit_ready_2024_gain_gt_5e6": submit_ready,
        })

    results.sort(key=lambda x: (x["gains_vs_v31"]["2024"], x["gains_vs_v31"]["2023"]), reverse=True)
    accepted_results = [r for r in results if r["all_seasons_improved"]]

    output = {
        "experiment": "V37b_context_smoothing",
        "description": "V35-strong form + context Trackman with varied smoothing",
        "form_config": FORM_CONFIG,
        "baseline_v31": {str(k): v for k, v in V31_BASELINE.items()},
        "accepted_count": len(accepted_results),
        "best": accepted_results[0] if accepted_results else None,
        "all_results": results,
    }
    Path("artifacts/v37_context_smoothing_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
