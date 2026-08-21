"""V54: Multi-Seed Averaging evaluation on V41 pipeline.

Retrain all 6 tree-based models (ExtraTrees, 3 Base HGBs, Form HGB, Context HGB)
with 5 different random seeds and average their OOF predictions.

This reduces model variance without changing any features, weights, or calibration.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_segment_calibration_v12 import segment_correction
from feature_engineering_v2 import LOW_CARDINAL_CATEGORICAL, add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


# V41 architecture constants
TREE_WEIGHTS = np.array([0.29483562599237795, 0.2344456574665245, 0.11617615437521091, 0.35454256216588664])
COMPONENTS = ["extra_trees", "trackman_hgb", "te_trackman_hgb", "hierarchical_hgb"]
W_V17 = 0.55
W_FORM = 0.32
W_CONTEXT = 0.13
W_TREES = 0.95
W_LOG = 0.05

FORM_CONFIG = {
    "max_leaf_nodes": 15, "min_samples_leaf": 200,
    "l2_regularization": 20.0, "learning_rate": 0.03, "max_iter": 500,
}

# V41 baseline Brier scores
V41_BASELINE = {
    2022: 0.24338879436541092,
    2023: 0.25315389703549307,
    2024: 0.24782554051940017,
}

SEEDS = [42, 1004, 2024, 777, 999]
MATCHUP_HTE = {"hte_pitcher_batter_100", "hte_pitcher_batter_500", "hte_pitcher_batter_log_count"}


def build_et_pipeline(frame, seed):
    categorical = [c for c in LOW_CARDINAL_CATEGORICAL if c in frame]
    numeric = [c for c in frame if c not in categorical]
    pipe = make_pipeline(
        ColumnTransformer([
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ), categorical),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ]),
        ExtraTreesClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=100,
            max_features=0.8, n_jobs=-1, random_state=seed,
        ),
    )
    return pipe, list(frame.columns)


def build_hgb_pipeline(frame, seed, config=None):
    categorical = [c for c in LOW_CARDINAL_CATEGORICAL if c in frame]
    numeric = [c for c in frame if c not in categorical]
    params = {
        "max_iter": 200, "learning_rate": 0.06, "max_leaf_nodes": 31,
        "min_samples_leaf": 100, "l2_regularization": 5.0, "random_state": seed,
    }
    if config:
        params.update(config)
        params["random_state"] = seed
    pipe = make_pipeline(
        ColumnTransformer([
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ), categorical),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ]),
        HistGradientBoostingClassifier(**params),
    )
    return pipe, list(frame.columns)


def no_matchup_columns(frame):
    return [c for c in frame.columns if c not in MATCHUP_HTE]


def main():
    print("Loading data...", flush=True)
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    raw_trackman = pd.read_csv("공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS)
    trackman = prepare_trackman(raw_trackman)
    context_trackman = prepare_context_trackman(trackman)

    # Precompute feature sets for each year
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")

    targets = {str(yr): oof[str(yr)]["target"].astype(float) for yr in (2022, 2023, 2024)}
    row_indices = {str(yr): oof[str(yr)]["row_index"] for yr in (2022, 2023, 2024)}
    frames = {str(yr): raw_frame.loc[row_indices[str(yr)]] for yr in (2022, 2023, 2024)}

    # Build feature sets per year (done once)
    year_features = {}
    for year in (2022, 2023, 2024):
        train_mask = data["season"] < year
        valid_mask = data["season"] == year
        prior = float(y.loc[train_mask].mean())

        base = select_v2_features(add_row_features(data, prior))
        base_tm = add_trackman_features(base, trackman)
        encoded_tm = add_trackman_features(
            select_v2_features(add_row_features(encoded, prior)), trackman
        )
        hier_tm = add_trackman_features(
            select_v2_features(add_row_features(hierarchical, prior)), trackman
        )
        form_raw = add_stable_form_features(hierarchical)
        form_base = select_v2_features(add_row_features(form_raw, prior))
        form_tm = add_trackman_features(form_base, trackman)
        form_cols = [c for c in form_tm.columns if not (c.startswith("tm_") and c.endswith("_std"))]

        ctx = add_context_trackman_features(hierarchical, context_trackman)
        ctx = add_trackman_features(ctx, trackman)
        ctx_base = select_v2_features(add_row_features(ctx, prior))
        ctx_cols = no_matchup_columns(ctx_base)

        year_features[year] = {
            "train_mask": train_mask,
            "valid_mask": valid_mask,
            "base": base,
            "base_tm": base_tm,
            "encoded_tm": encoded_tm,
            "hier_tm": hier_tm,
            "form_tm": form_tm,
            "form_cols": form_cols,
            "ctx_base": ctx_base,
            "ctx_cols": ctx_cols,
        }
    print("Feature sets prepared.", flush=True)

    # Train with multiple seeds and collect OOF predictions
    seed_predictions = {seed: {} for seed in SEEDS}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===", flush=True)

        for year in (2022, 2023, 2024):
            f = year_features[year]
            train_mask = f["train_mask"]
            valid_mask = f["valid_mask"]

            # 1. ExtraTrees on base features
            et_pipe, et_cols = build_et_pipeline(f["base"], seed)
            et_pipe.fit(f["base"].loc[train_mask, et_cols], y.loc[train_mask])
            et_pred = et_pipe.predict_proba(f["base"].loc[valid_mask, et_cols])[:, 1]

            # 2. Trackman HGB on base+trackman
            tm_pipe, tm_cols = build_hgb_pipeline(f["base_tm"], seed)
            tm_pipe.fit(f["base_tm"].loc[train_mask, tm_cols], y.loc[train_mask])
            tm_pred = tm_pipe.predict_proba(f["base_tm"].loc[valid_mask, tm_cols])[:, 1]

            # 3. Encoded HGB on encoded+trackman
            enc_pipe, enc_cols = build_hgb_pipeline(f["encoded_tm"], seed)
            enc_pipe.fit(f["encoded_tm"].loc[train_mask, enc_cols], y.loc[train_mask])
            enc_pred = enc_pipe.predict_proba(f["encoded_tm"].loc[valid_mask, enc_cols])[:, 1]

            # 4. Hierarchical HGB on hierarchical+trackman
            hier_pipe, hier_cols = build_hgb_pipeline(f["hier_tm"], seed)
            hier_pipe.fit(f["hier_tm"].loc[train_mask, hier_cols], y.loc[train_mask])
            hier_pred = hier_pipe.predict_proba(f["hier_tm"].loc[valid_mask, hier_cols])[:, 1]

            # V11 tree blend
            tree_pred = (TREE_WEIGHTS[0] * et_pred + TREE_WEIGHTS[1] * tm_pred +
                        TREE_WEIGHTS[2] * enc_pred + TREE_WEIGHTS[3] * hier_pred)

            # V17 = 0.95 * trees + 0.05 * logistic (logistic is fixed, no randomness)
            v17_pred = W_TREES * tree_pred + W_LOG * logistic[str(year)]

            # 5. Form HGB (gentle_500)
            form_pipe, form_cols = build_hgb_pipeline(
                f["form_tm"][f["form_cols"]], seed, FORM_CONFIG
            )
            form_pipe.fit(f["form_tm"].loc[train_mask, form_cols], y.loc[train_mask])
            form_pred = form_pipe.predict_proba(f["form_tm"].loc[valid_mask, form_cols])[:, 1]

            # 6. Context HGB (V31 default, no matchup HTE)
            ctx_pipe, ctx_cols = build_hgb_pipeline(f["ctx_base"][f["ctx_cols"]], seed)
            ctx_pipe.fit(f["ctx_base"].loc[train_mask, ctx_cols], y.loc[train_mask])
            ctx_pred = ctx_pipe.predict_proba(f["ctx_base"].loc[valid_mask, ctx_cols])[:, 1]

            # 3-way blend
            raw_blend = W_V17 * v17_pred + W_FORM * form_pred + W_CONTEXT * ctx_pred

            seed_predictions[seed][str(year)] = raw_blend
            print(f"  year={year} done", flush=True)

    # Average predictions across seeds
    avg_predictions = {}
    for year in (2022, 2023, 2024):
        preds = np.stack([seed_predictions[s][str(year)] for s in SEEDS])
        avg_predictions[str(year)] = preds.mean(axis=0)

    # Evaluate: single-seed (seed=42) vs multi-seed average
    f23_train_frame = frames["2022"]
    f24_train_frame = pd.concat([frames["2022"], frames["2023"]])

    results = {}
    for label, preds in [("single_seed_42", seed_predictions[42]), ("multi_seed_avg", avg_predictions)]:
        score22 = float(brier_score_loss(targets["2022"], preds["2022"]))

        res22 = targets["2022"] - preds["2022"]
        cnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["balls_before", "strikes_before"], 500)
        pcnt_23 = segment_correction(f23_train_frame, res22, frames["2023"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred23_cal = np.clip(preds["2023"] + res22.mean() + 0.75 * cnt_23 + 0.25 * pcnt_23, 0, 1)
        score23 = float(brier_score_loss(targets["2023"], pred23_cal))

        res_22_23 = np.concatenate([res22, targets["2023"] - preds["2023"]])
        cnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["balls_before", "strikes_before"], 500)
        pcnt_24 = segment_correction(f24_train_frame, res_22_23, frames["2024"], ["pitcher_id", "balls_before", "strikes_before"], 300)
        pred24_cal = np.clip(preds["2024"] + res_22_23.mean() + 0.75 * cnt_24 + 0.25 * pcnt_24, 0, 1)
        score24 = float(brier_score_loss(targets["2024"], pred24_cal))

        gain22 = V41_BASELINE[2022] - score22
        gain23 = V41_BASELINE[2023] - score23
        gain24 = V41_BASELINE[2024] - score24

        results[label] = {
            "scores": {"2022": score22, "2023": score23, "2024": score24},
            "gains_vs_v41": {"2022": gain22, "2023": gain23, "2024": gain24},
            "all_improved": (gain22 > 0 and gain23 > 0 and gain24 > 0),
            "submit_ready": (gain22 > 0 and gain23 > 0 and gain24 > 5e-6),
        }
        print(f"\n{label}: gains 2022={gain22:.8f}, 2023={gain23:.8f}, 2024={gain24:.8f}", flush=True)

    output = {
        "experiment": "V54_multi_seed_averaging",
        "description": "Train all 6 tree models with 5 seeds (42,1004,2024,777,999) and average predictions",
        "seeds": SEEDS,
        "baseline_v41": {str(k): v for k, v in V41_BASELINE.items()},
        "results": results,
    }
    Path("artifacts/v54_multi_seed_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
