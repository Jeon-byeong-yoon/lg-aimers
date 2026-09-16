# script.py — Evaluation server inference script (v3)
# LightGBM + CatBoost ensemble with Isotonic Calibration
import os
import json
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoost

DATA_DIR   = "./data"
MODEL_DIR  = "./model"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ID_COL, TARGET_COL = "row_id", "control_success"
N_FOLDS = 5

def engineer_features(df):
    df = df.copy()

    base = 'asof_pitcher_success_rate'
    p1 = 'asof_pitcher_prev1_game_success_rate'
    p3 = 'asof_pitcher_prev3_game_success_rate'
    p5 = 'asof_pitcher_prev5_game_success_rate'
    
    if p1 in df and base in df: df['feat_form_trend_1'] = df[p1] - df[base]
    if p3 in df and base in df: df['feat_form_trend_3'] = df[p3] - df[base]
    if p5 in df and base in df: df['feat_form_trend_5'] = df[p5] - df[base]
    if p1 in df and p3 in df:   df['feat_form_prev1_vs_3'] = df[p1] - df[p3]
    if p3 in df and p5 in df:   df['feat_form_prev3_vs_5'] = df[p3] - df[p5]
    
    fm = {p1: 0.5, p3: 0.3, p5: 0.2}
    av = {k: v for k, v in fm.items() if k in df}
    if av and base in df:
        df['feat_weighted_form'] = sum(df[c].fillna(df[base].fillna(0.5)) * w for c, w in av.items()) / sum(av.values())

    m_base = 'asof_pitcher_middle_rate'
    mp1 = 'asof_pitcher_prev1_game_middle_rate'
    mp3 = 'asof_pitcher_prev3_game_middle_rate'
    mp5 = 'asof_pitcher_prev5_game_middle_rate'
    if mp1 in df and m_base in df: df['feat_middle_trend_1'] = df[mp1] - df[m_base]
    if mp3 in df and m_base in df: df['feat_middle_trend_3'] = df[mp3] - df[m_base]
    if mp5 in df and m_base in df: df['feat_middle_trend_5'] = df[mp5] - df[m_base]

    if 'asof_pitcher_success_rate' in df and 'asof_batter_success_rate' in df:
        df['feat_pitcher_batter_edge'] = df['asof_pitcher_success_rate'] - df['asof_batter_success_rate']
    if 'asof_pitcher_middle_rate' in df and 'asof_batter_middle_rate' in df:
        df['feat_pitcher_batter_middle_edge'] = df['asof_pitcher_middle_rate'] - df['asof_batter_middle_rate']
    if 'pitcher_hand' in df and 'batter_hand' in df:
        df['feat_platoon_same'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)
        df['feat_matchup_code'] = df['pitcher_hand'] * 10 + df['batter_hand']

    if 'balls_before' in df and 'strikes_before' in df:
        df['feat_count_pressure'] = df['balls_before'] - df['strikes_before']
        df['feat_count_code']     = df['balls_before'] * 10 + df['strikes_before']
        df['feat_full_count']     = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(int)
        df['feat_ahead_count']    = (df['strikes_before'] > df['balls_before']).astype(int)
        df['feat_behind_count']   = (df['balls_before'] > df['strikes_before']).astype(int)

    runners = [c for c in ['runner_on_1b', 'runner_on_2b', 'runner_on_3b'] if c in df]
    if runners:
        df['feat_runners_total'] = df[runners].sum(axis=1)
        if 'runner_on_2b' in df and 'runner_on_3b' in df:
            df['feat_scoring_position'] = ((df['runner_on_2b'] == 1) | (df['runner_on_3b'] == 1)).astype(int)
    if 'num_runners_on' in df:
        df['feat_bases_loaded'] = (df['num_runners_on'] == 3).astype(int)
    if 'li' in df:
        df['feat_log_li'] = np.log1p(df['li'])
        if 'feat_runners_total' in df:
            df['feat_leverage_pressure'] = df['li'] * df['feat_runners_total']
    if 'inning' in df and 'score_diff_pitcher_team' in df:
        df['feat_late_close'] = ((df['inning'] >= 7) & (df['score_diff_pitcher_team'].abs() <= 2)).astype(int)

    if 'asof_pitcher_strike_rate' in df and 'asof_pitcher_ball_rate' in df:
        df['feat_strike_ball_diff'] = df['asof_pitcher_strike_rate'] - df['asof_pitcher_ball_rate']
    if 'asof_pitcher_reverse_rate' in df and 'asof_pitcher_middle_rate' in df:
        df['feat_danger_zone_rate'] = df['asof_pitcher_reverse_rate'] + df['asof_pitcher_middle_rate']

    return df

def isotonic_predict(x, iso_x, iso_y):
    x = np.clip(np.asarray(x, dtype=float), iso_x[0], iso_x[-1])
    idx = np.searchsorted(iso_x, x, side="right") - 1
    idx = np.clip(idx, 0, len(iso_y) - 1)
    return iso_y[idx]

def main():
    print("Load config...")
    with open(os.path.join(MODEL_DIR, "feature_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    FEATURES     = cfg["features"]
    ALL_CAT_COLS = cfg["cat_cols"]
    median_vals  = pd.Series(cfg["median_vals"])

    with open(os.path.join(MODEL_DIR, "le_mappings.json"), "r", encoding="utf-8") as f:
        le_mappings = json.load(f)

    iso_x = np.load(os.path.join(MODEL_DIR, "iso_x.npy"))
    iso_y = np.load(os.path.join(MODEL_DIR, "iso_y.npy"))
    opt_w = np.load(os.path.join(MODEL_DIR, "opt_weights.npy"))

    print("Load test data...")
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    sub  = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")

    print("Feature engineering...")
    test = engineer_features(test)

    # Categorical Label Encoding
    for col in ALL_CAT_COLS:
        if col in test.columns and col in le_mappings:
            mapping = le_mappings[col]
            test[col] = test[col].astype(str).map(lambda x, m=mapping: m.get(x, 0)).astype(int)

    feat_cols = [c for c in FEATURES if c in test.columns]
    X_test = test[feat_cols].copy()

    num_cols = [c for c in feat_cols if c not in ALL_CAT_COLS]
    X_test[num_cols] = X_test[num_cols].fillna(median_vals.reindex(num_cols))

    print("Inference...")
    lgb_p = np.zeros(len(X_test))
    cat_p = np.zeros(len(X_test))

    w_lgb = float(opt_w[0])
    w_cat = float(opt_w[1])

    for fold in range(N_FOLDS):
        # LightGBM Booster
        m_lgb = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lgb_fold{fold}.txt"))
        lgb_p += m_lgb.predict(X_test) / N_FOLDS

        # CatBoost
        m_cat = CatBoost()
        m_cat.load_model(os.path.join(MODEL_DIR, f"cat_fold{fold}.cbm"))
        cat_p += m_cat.predict(X_test, prediction_type="Probability")[:, 1] / N_FOLDS

    blend = w_lgb * lgb_p + w_cat * cat_p

    # Isotonic calibration
    final = isotonic_predict(blend, iso_x, iso_y)
    final = np.clip(final, 0.0, 1.0)

    if any(not math.isfinite(float(p)) or not 0.0 <= float(p) <= 1.0 for p in final):
        raise ValueError("Invalid prediction output")

    pred_map = dict(zip(test[ID_COL].tolist(), final.tolist()))
    sub[TARGET_COL] = sub[ID_COL].map(pred_map)

    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path} (rows={len(sub)}, mean={final.mean():.4f})")

if __name__ == "__main__":
    main()
