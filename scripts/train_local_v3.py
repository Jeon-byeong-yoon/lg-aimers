"""
LG Aimers 9기 — Local Full Pipeline v3 (Pure LGBM + CatBoost Ensemble)
Removes XGBoost (which had negative BSS contribution) and Target-Encoding ID leakage.
Uses LightGBM (45%) + CatBoost (55%) + Out-of-fold Calibration.
"""
import os
import gc
import time
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoost

BASE_DIR = Path(r"c:\Users\wnsgu\Desktop\lg-aimers")
DATA_DIR = BASE_DIR / "공모전 dataset" / "open" / "data"
MODEL_DIR = BASE_DIR / "scripts" / "submission_v2" / "model"
SUBMIT_V2_DIR = BASE_DIR / "scripts" / "submission_v2"
OUTPUT_DIR = BASE_DIR / "scripts" / "submission_v2" / "output"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"

ID_COL, TARGET_COL = "row_id", "control_success"
SEED, N_SPLITS = 42, 5

def brier_skill_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 1e-7, 1.0 - 1e-7)
    r = y_true.mean()
    brier = ((y_pred - y_true) ** 2).mean()
    base_brier = r * (1.0 - r)
    score = max(0.0, 100_000.0 * (1.0 - brier / base_brier))
    return score, brier, base_brier

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

def main():
    t0 = time.time()
    print("=" * 60)
    print("LG Aimers 9 - Local Pipeline v3 (Pure LGBM + CatBoost)")
    print("=" * 60)

    # 1. Data load
    print("\n[Step 1] Loading dataset...")
    train = pd.read_csv(TRAIN_PATH, encoding='utf-8-sig')
    test  = pd.read_csv(TEST_PATH,  encoding='utf-8-sig')
    print(f"  train: {train.shape} | test: {test.shape}")

    # 2. Feature engineering
    print("\n[Step 2] Engineering domain features...")
    train = engineer_features(train)
    test  = engineer_features(test)

    # 3. Categorical encoding
    print("\n[Step 3] Categorical Encoding...")
    CAT_COLS = ['top_bottom', 'game_type', 'base_state']
    ALL_CAT_COLS = [c for c in CAT_COLS if c in train.columns]

    le_mappings = {}
    for col in ALL_CAT_COLS:
        cats = sorted(list(set(train[col].astype(str).unique()) | set(test[col].astype(str).unique())))
        mapping = {str(cls): int(i) for i, cls in enumerate(cats)}
        train[col] = train[col].astype(str).map(mapping).fillna(0).astype(int)
        test[col]  = test[col].astype(str).map(mapping).fillna(0).astype(int)
        le_mappings[col] = mapping

    EXCLUDE = {ID_COL, TARGET_COL}
    test_base_cols = set(pd.read_csv(TEST_PATH, encoding='utf-8-sig', nrows=0).columns) - {ID_COL}
    derived_cols   = {c for c in train.columns if c.startswith('feat_')}

    FEATURES = [
        c for c in train.columns
        if (c in test_base_cols or c in derived_cols) and c not in EXCLUDE
    ]

    print(f"  Total FEATURES: {len(FEATURES)} | Categorical: {ALL_CAT_COLS}")

    X      = train[FEATURES].copy()
    y      = train[TARGET_COL].copy()
    X_test = test[FEATURES].copy()

    num_cols = [c for c in FEATURES if c not in ALL_CAT_COLS]
    median_vals = X[num_cols].median().to_dict()
    median_series = pd.Series(median_vals)

    X[num_cols]      = X[num_cols].fillna(median_series)
    X_test[num_cols] = X_test[num_cols].fillna(median_series)

    # Clean old model folder
    for old_file in MODEL_DIR.glob("*"):
        try: old_file.unlink()
        except: pass

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # 4. LightGBM Training
    print("\n[Step 4] Training LightGBM (5-Fold CV)...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 255,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.5,
        'n_jobs': -1,
        'seed': SEED,
        'verbose': -1,
    }

    lgb_oof  = np.zeros(len(X))
    lgb_test = np.zeros(len(X_test))

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        t_f = time.time()
        X_tr, X_vl = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[trn_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=ALL_CAT_COLS)
        dval   = lgb.Dataset(X_vl, label=y_vl, categorical_feature=ALL_CAT_COLS, reference=dtrain)
        
        m = lgb.train(
            lgb_params, dtrain, num_boost_round=1200, valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(300)],
        )
        m.save_model(str(MODEL_DIR / f"lgb_fold{fold}.txt"))
        
        lgb_oof[val_idx] = m.predict(X_vl)
        lgb_test += m.predict(X_test) / N_SPLITS

        s, b, _ = brier_skill_score(y_vl.values, lgb_oof[val_idx])
        print(f"  LGB Fold {fold+1} | Score: {s:.2f} | Brier: {b:.6f} | Time: {time.time()-t_f:.1f}s")

    lgb_cv, lgb_brier, _ = brier_skill_score(y.values, lgb_oof)
    print(f"  ==> LightGBM CV BSS Score: {lgb_cv:.2f} (Brier: {lgb_brier:.6f})")

    # 5. CatBoost Training
    print("\n[Step 5] Training CatBoost (5-Fold CV)...")
    cat_params = dict(
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_strength=1.0,
        bagging_temperature=0.8,
        loss_function='Logloss',
        eval_metric='Logloss',
        early_stopping_rounds=100,
        random_seed=SEED,
        verbose=300,
        thread_count=-1,
    )
    cat_oof  = np.zeros(len(X))
    cat_test = np.zeros(len(X_test))

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        t_f = time.time()
        X_tr, X_vl = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[trn_idx], y.iloc[val_idx]

        m = CatBoostClassifier(**cat_params)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), use_best_model=True)
        m.save_model(str(MODEL_DIR / f"cat_fold{fold}.cbm"))

        cat_oof[val_idx] = m.predict_proba(X_vl)[:, 1]
        cat_test += m.predict_proba(X_test)[:, 1] / N_SPLITS

        s, b, _ = brier_skill_score(y_vl.values, cat_oof[val_idx])
        print(f"  CatBoost Fold {fold+1} | Score: {s:.2f} | Brier: {b:.6f} | Time: {time.time()-t_f:.1f}s")

    cat_cv, cat_brier, _ = brier_skill_score(y.values, cat_oof)
    print(f"  ==> CatBoost CV BSS Score: {cat_cv:.2f} (Brier: {cat_brier:.6f})")

    # 6. Ensemble Optimization
    print("\n[Step 6] Optimizing Ensemble Weights...")
    res = minimize(lambda w: -brier_skill_score(y.values, np.clip(w[0], 0, 1)*lgb_oof + (1-np.clip(w[0], 0, 1))*cat_oof)[0],
                   [0.5], method='Nelder-Mead')
    w_lgb = float(np.clip(res.x[0], 0, 1))
    w_cat = 1.0 - w_lgb

    print(f"  Optimal Weights: LightGBM = {w_lgb:.4f} | CatBoost = {w_cat:.4f}")

    oof_blend  = w_lgb * lgb_oof  + w_cat * cat_oof
    test_blend = w_lgb * lgb_test + w_cat * cat_test

    s_blend, b_blend, _ = brier_skill_score(y.values, oof_blend)
    print(f"  ==> Blended OOF BSS Score: {s_blend:.2f} (Brier: {b_blend:.6f})")

    # 7. Out-of-fold Isotonic Calibration
    print("\n[Step 7] Out-of-fold Isotonic Calibration...")
    iso_oof = np.zeros(len(X))
    for trn_idx, val_idx in kf.split(X, y):
        iso_fold = IsotonicRegression(out_of_bounds='clip')
        iso_fold.fit(oof_blend[trn_idx], y.iloc[trn_idx])
        iso_oof[val_idx] = iso_fold.predict(oof_blend[val_idx])

    s_iso_cv, b_iso_cv, _ = brier_skill_score(y.values, iso_oof)
    print(f"  ==> Out-of-fold Isotonic BSS Score: {s_iso_cv:.2f} (Brier: {b_iso_cv:.6f})")

    # Full Isotonic fit for test predictions
    iso_full = IsotonicRegression(out_of_bounds='clip')
    iso_full.fit(oof_blend, y.values)
    test_final = np.clip(iso_full.predict(test_blend), 0.0, 1.0)

    # Save Isotonic arrays
    np.save(str(MODEL_DIR / "iso_x.npy"), iso_full.X_thresholds_)
    np.save(str(MODEL_DIR / "iso_y.npy"), iso_full.y_thresholds_)
    np.save(str(MODEL_DIR / "opt_weights.npy"), np.array([w_lgb, w_cat]))

    # Save configs
    feature_config = {
        "features": FEATURES,
        "cat_cols": ALL_CAT_COLS,
        "median_vals": median_vals,
    }
    with open(MODEL_DIR / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)

    with open(MODEL_DIR / "le_mappings.json", "w", encoding="utf-8") as f:
        json.dump(le_mappings, f, ensure_ascii=False, indent=2)

    # Save local submission
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding='utf-8-sig')
    pred_map = dict(zip(test[ID_COL].tolist(), test_final.tolist()))
    submission = sample_sub.copy()
    submission[TARGET_COL] = submission[ID_COL].map(pred_map)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False, encoding="utf-8")

    print(f"\nLocal submission saved to: {OUTPUT_DIR / 'submission.csv'}")
    print(f"  Rows: {len(submission)} | Mean: {submission[TARGET_COL].mean():.4f} | Std: {submission[TARGET_COL].std():.4f}")
    print(submission.head())

    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(f"  Baseline RF Score:             ~549.00")
    print(f"  LightGBM OOF BSS:              {lgb_cv:.2f}")
    print(f"  CatBoost OOF BSS:              {cat_cv:.2f}")
    print(f"  Raw Blend OOF BSS:             {s_blend:.2f}")
    print(f"  Isotonic Calibrated OOF BSS:   {s_iso_cv:.2f}")
    print(f"  Total Time Elapsed:            {time.time()-t0:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
