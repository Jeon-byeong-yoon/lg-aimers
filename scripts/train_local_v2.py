"""
LG Aimers 9기 — Local Full Pipeline (v2)
LightGBM + CatBoost + XGBoost Ensemble with OOF Target Encoding & Platt Scaling
"""
import os
import sys
import gc
import time
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoost

# ── 0. 경로 설정 ──────────────────────────────────────────────────
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

# ── 1. Brier Skill Score 함수 ────────────────────────────────────
def brier_skill_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 1e-7, 1 - 1e-7)
    r = y_true.mean()
    brier = ((y_pred - y_true) ** 2).mean()
    base_brier = r * (1.0 - r)
    score = max(0.0, 100_000.0 * (1.0 - brier / base_brier))
    return score, brier, base_brier

# ── 2. 피처 엔지니어링 ───────────────────────────────────────────
def engineer_features(df):
    print("  [FE] Adding domain features...")
    df = df.copy()

    base = 'asof_pitcher_success_rate'
    p1 = 'asof_pitcher_prev1_game_success_rate'
    p3 = 'asof_pitcher_prev3_game_success_rate'
    p5 = 'asof_pitcher_prev5_game_success_rate'
    
    # Form trends & weighted form
    if p1 in df and base in df: df['feat_form_trend_1'] = df[p1] - df[base]
    if p3 in df and base in df: df['feat_form_trend_3'] = df[p3] - df[base]
    if p5 in df and base in df: df['feat_form_trend_5'] = df[p5] - df[base]
    if p1 in df and p3 in df:   df['feat_form_prev1_vs_3'] = df[p1] - df[p3]
    if p3 in df and p5 in df:   df['feat_form_prev3_vs_5'] = df[p3] - df[p5]
    
    fm = {p1: 0.5, p3: 0.3, p5: 0.2}
    av = {k: v for k, v in fm.items() if k in df}
    if av and base in df:
        df['feat_weighted_form'] = sum(df[c].fillna(df[base].fillna(0.5)) * w for c, w in av.items()) / sum(av.values())

    # Middle trends
    m_base = 'asof_pitcher_middle_rate'
    mp1 = 'asof_pitcher_prev1_game_middle_rate'
    mp3 = 'asof_pitcher_prev3_game_middle_rate'
    mp5 = 'asof_pitcher_prev5_game_middle_rate'
    if mp1 in df and m_base in df: df['feat_middle_trend_1'] = df[mp1] - df[m_base]
    if mp3 in df and m_base in df: df['feat_middle_trend_3'] = df[mp3] - df[m_base]
    if mp5 in df and m_base in df: df['feat_middle_trend_5'] = df[mp5] - df[m_base]

    # Pitcher-batter matchup & edge
    if 'asof_pitcher_success_rate' in df and 'asof_batter_success_rate' in df:
        df['feat_pitcher_batter_edge'] = df['asof_pitcher_success_rate'] - df['asof_batter_success_rate']
    if 'asof_pitcher_middle_rate' in df and 'asof_batter_middle_rate' in df:
        df['feat_pitcher_batter_middle_edge'] = df['asof_pitcher_middle_rate'] - df['asof_batter_middle_rate']
    if 'pitcher_hand' in df and 'batter_hand' in df:
        df['feat_platoon_same'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)
        df['feat_matchup_code'] = df['pitcher_hand'] * 10 + df['batter_hand']

    # Count features
    if 'balls_before' in df and 'strikes_before' in df:
        df['feat_count_pressure'] = df['balls_before'] - df['strikes_before']
        df['feat_count_code']     = df['balls_before'] * 10 + df['strikes_before']
        df['feat_full_count']     = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(int)
        df['feat_ahead_count']    = (df['strikes_before'] > df['balls_before']).astype(int)
        df['feat_behind_count']   = (df['balls_before'] > df['strikes_before']).astype(int)
        df['feat_two_strikes']    = (df['strikes_before'] == 2).astype(int)
        df['feat_three_balls']    = (df['balls_before'] == 3).astype(int)

    # Baserunners & leverage
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
        df['feat_score_diff_abs'] = df['score_diff_pitcher_team'].abs()

    # Ratios & Workload
    if 'asof_pitcher_strike_rate' in df and 'asof_pitcher_ball_rate' in df:
        df['feat_strike_ball_diff'] = df['asof_pitcher_strike_rate'] - df['asof_pitcher_ball_rate']
    if 'asof_pitcher_reverse_rate' in df and 'asof_pitcher_middle_rate' in df:
        df['feat_danger_zone_rate'] = df['asof_pitcher_reverse_rate'] + df['asof_pitcher_middle_rate']
    if 'asof_pitcher_n' in df:
        df['feat_log_asof_pitcher_n'] = np.log1p(df['asof_pitcher_n'])
        df['feat_cold_start_pitcher'] = (df['asof_pitcher_n'] < 30).astype(int)
    if 'asof_batter_n' in df:
        df['feat_log_asof_batter_n'] = np.log1p(df['asof_batter_n'])
        df['feat_cold_start_batter'] = (df['asof_batter_n'] < 30).astype(int)

    return df

# ── 3. 메인 실행 ─────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("LG Aimers 9 - Local Full Training Pipeline v2")
    print("=" * 60)

    # [A] 데이터 로드
    print("\n[Step 1] Loading train.csv and test.csv...")
    train = pd.read_csv(TRAIN_PATH, encoding='utf-8-sig')
    test  = pd.read_csv(TEST_PATH,  encoding='utf-8-sig')
    print(f"  train: {train.shape} | test: {test.shape}")
    print(f"  Target control_success mean: {train[TARGET_COL].mean():.4f}")

    # [B] 피처 엔지니어링
    print("\n[Step 2] Feature Engineering...")
    train = engineer_features(train)
    test  = engineer_features(test)

    # [C] Target Encoding 설정
    print("\n[Step 3] Out-of-fold Target Encoding...")
    cols_to_encode = [
        ('pitcher_id', 20),
        ('batter_id', 30),
        ('pitcher_team_id', 100),
        ('batter_team_id', 100),
        ('base_state', 50),
        ('feat_count_code', 50),
    ]

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    global_mean = float(train[TARGET_COL].mean())

    te_features = []
    target_encoders = {"global_mean": global_mean, "mappings": {}}

    for col, m in cols_to_encode:
        te_col_name = f"{col}_te"
        te_features.append(te_col_name)
        train[te_col_name] = global_mean
        test[te_col_name]  = global_mean

        for trn_idx, val_idx in kf.split(train, train[TARGET_COL]):
            tr_df = train.iloc[trn_idx]
            stats = tr_df.groupby(col)[TARGET_COL].agg(['count', 'mean'])
            smoothed = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
            mapping = smoothed.to_dict()
            train.iloc[val_idx, train.columns.get_loc(te_col_name)] = (
                train.iloc[val_idx][col].map(mapping).fillna(global_mean)
            )

        # Full dataset mapping for test set & saving lookup dict
        stats_full = train.groupby(col)[TARGET_COL].agg(['count', 'mean'])
        smoothed_full = (stats_full['count'] * stats_full['mean'] + m * global_mean) / (stats_full['count'] + m)
        mapping_full = {str(k): float(v) for k, v in smoothed_full.to_dict().items()}
        test[te_col_name] = test[col].astype(str).map(mapping_full).fillna(global_mean)
        target_encoders["mappings"][col] = {"m": m, "map": mapping_full}

    print(f"  Added {len(te_features)} Target Encoded features")

    # [D] 피처 확정 & Categorical Encoding
    print("\n[Step 4] Categorical Encoding & Feature Selection...")
    EXCLUDE = {ID_COL, TARGET_COL}
    CAT_COLS = ['top_bottom', 'game_type', 'base_state']
    ALL_CAT_COLS = [c for c in CAT_COLS if c in train.columns]

    le_mappings = {}
    for col in ALL_CAT_COLS:
        categories = sorted(list(set(train[col].astype(str).unique()) | set(test[col].astype(str).unique())))
        mapping = {str(cls): int(i) for i, cls in enumerate(categories)}
        train[col] = train[col].astype(str).map(mapping).fillna(0).astype(int)
        test[col]  = test[col].astype(str).map(mapping).fillna(0).astype(int)
        le_mappings[col] = mapping

    test_base_cols = set(pd.read_csv(TEST_PATH, encoding='utf-8-sig', nrows=0).columns) - {ID_COL}
    derived_cols   = {c for c in train.columns if c.startswith('feat_') or c.endswith('_te')}

    FEATURES = [
        c for c in train.columns
        if (c in test_base_cols or c in derived_cols) and c not in EXCLUDE
    ]

    print(f"  Total FEATURES: {len(FEATURES)} | Categorical: {ALL_CAT_COLS}")

    X      = train[FEATURES].copy()
    y      = train[TARGET_COL].copy()
    X_test = test[FEATURES].copy()

    # Impute missing values with medians
    num_cols = [c for c in FEATURES if c not in ALL_CAT_COLS]
    median_vals = X[num_cols].median().to_dict()
    median_series = pd.Series(median_vals)

    X[num_cols]      = X[num_cols].fillna(median_series)
    X_test[num_cols] = X_test[num_cols].fillna(median_series)

    print(f"  X shape: {X.shape} | X_test shape: {X_test.shape}")

    # [E] LightGBM Training
    print("\n[Step 5] Training LightGBM (5-Fold CV)...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.03,
        'num_leaves': 127,
        'max_depth': -1,
        'min_child_samples': 100,
        'feature_fraction': 0.7,
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
        
        # Save native model file
        m.save_model(str(MODEL_DIR / f"lgb_fold{fold}.txt"))
        
        lgb_oof[val_idx] = m.predict(X_vl)
        lgb_test += m.predict(X_test) / N_SPLITS

        s, b, _ = brier_skill_score(y_vl.values, lgb_oof[val_idx])
        print(f"  LGB Fold {fold+1} | Score: {s:.2f} | Brier: {b:.6f} | Time: {time.time()-t_f:.1f}s")

    lgb_cv, lgb_brier, _ = brier_skill_score(y.values, lgb_oof)
    print(f"  ==> LightGBM CV BSS Score: {lgb_cv:.2f} (Brier: {lgb_brier:.6f})")

    # [F] CatBoost Training
    print("\n[Step 6] Training CatBoost (5-Fold CV)...")
    cat_params = dict(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=5,
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

    # [G] XGBoost Training
    print("\n[Step 7] Training XGBoost (5-Fold CV)...")
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.03,
        'max_depth': 6,
        'min_child_weight': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'n_estimators': 1200,
        'random_state': SEED,
        'n_jobs': -1,
        'early_stopping_rounds': 100,
    }
    xgb_oof  = np.zeros(len(X))
    xgb_test = np.zeros(len(X_test))

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        t_f = time.time()
        X_tr, X_vl = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[trn_idx], y.iloc[val_idx]

        m = xgb.XGBClassifier(**xgb_params)
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=300)
        m.save_model(str(MODEL_DIR / f"xgb_fold{fold}.json"))

        xgb_oof[val_idx] = m.predict_proba(X_vl)[:, 1]
        xgb_test += m.predict_proba(X_test)[:, 1] / N_SPLITS

        s, b, _ = brier_skill_score(y_vl.values, xgb_oof[val_idx])
        print(f"  XGBoost Fold {fold+1} | Score: {s:.2f} | Brier: {b:.6f} | Time: {time.time()-t_f:.1f}s")

    xgb_cv, xgb_brier, _ = brier_skill_score(y.values, xgb_oof)
    print(f"  ==> XGBoost CV BSS Score: {xgb_cv:.2f} (Brier: {xgb_brier:.6f})")

    # [H] Ensemble Optimization
    print("\n[Step 8] Optimizing Ensemble Weights...")
    def neg_bss(weights):
        w = np.clip(weights, 0, 1)
        if w.sum() == 0:
            return 0.0
        w = w / w.sum()
        blend = w[0]*lgb_oof + w[1]*cat_oof + w[2]*xgb_oof
        s, _, _ = brier_skill_score(y.values, blend)
        return -s

    res = minimize(neg_bss, [1/3, 1/3, 1/3], method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-7})
    opt_w = np.clip(res.x, 0, 1)
    opt_w /= opt_w.sum()
    print(f"  Optimal Weights: LightGBM={opt_w[0]:.4f} | CatBoost={opt_w[1]:.4f} | XGBoost={opt_w[2]:.4f}")

    oof_blend  = opt_w[0]*lgb_oof  + opt_w[1]*cat_oof  + opt_w[2]*xgb_oof
    test_blend = opt_w[0]*lgb_test + opt_w[1]*cat_test + opt_w[2]*xgb_test

    s_blend, b_blend, _ = brier_skill_score(y.values, oof_blend)
    print(f"  ==> Blended OOF BSS Score: {s_blend:.2f} (Brier: {b_blend:.6f})")

    # [I] Platt Scaling (Logistic Regression on Log-odds)
    print("\n[Step 9] Fitting Smooth Platt Scaling (Probability Calibration)...")
    eps = 1e-7
    oof_logits = np.log(np.clip(oof_blend, eps, 1 - eps) / np.clip(1 - oof_blend, eps, 1 - eps)).reshape(-1, 1)
    test_logits = np.log(np.clip(test_blend, eps, 1 - eps) / np.clip(1 - test_blend, eps, 1 - eps)).reshape(-1, 1)

    platt = LogisticRegression(C=1.0, solver='lbfgs')
    platt.fit(oof_logits, y.values)

    a = float(platt.coef_[0][0])
    b = float(platt.intercept_[0])
    print(f"  Platt Calibration Parameters: a = {a:.6f}, b = {b:.6f}")

    oof_cal  = 1.0 / (1.0 + np.exp(-(a * oof_logits.ravel() + b)))
    test_cal = 1.0 / (1.0 + np.exp(-(a * test_logits.ravel() + b)))

    s_cal, b_cal, _ = brier_skill_score(y.values, oof_cal)
    print(f"  ==> Calibrated OOF BSS Score: {s_cal:.2f} (Brier: {b_cal:.6f})")

    # [J] Save Artifacts
    print("\n[Step 10] Saving All Configuration & Model Artifacts...")
    
    # Save ensemble weights
    np.save(str(MODEL_DIR / "opt_weights.npy"), opt_w)

    # Save feature config
    feature_config = {
        "features": FEATURES,
        "cat_cols": ALL_CAT_COLS,
        "median_vals": median_vals,
    }
    with open(MODEL_DIR / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)

    # Save label encoder mappings
    with open(MODEL_DIR / "le_mappings.json", "w", encoding="utf-8") as f:
        json.dump(le_mappings, f, ensure_ascii=False, indent=2)

    # Save target encoders dict
    with open(MODEL_DIR / "target_encoders.json", "w", encoding="utf-8") as f:
        json.dump(target_encoders, f, ensure_ascii=False, indent=2)

    # Save Platt calibrator parameters
    platt_config = {"a": a, "b": b}
    with open(MODEL_DIR / "platt_calibrator.json", "w", encoding="utf-8") as f:
        json.dump(platt_config, f, ensure_ascii=False, indent=2)

    # Generate local submission for verification
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding='utf-8-sig')
    pred_map = dict(zip(test[ID_COL].tolist(), test_cal.tolist()))
    submission = sample_sub.copy()
    submission[TARGET_COL] = submission[ID_COL].map(pred_map)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False, encoding="utf-8")

    print(f"  Local submission saved to: {OUTPUT_DIR / 'submission.csv'}")
    print(f"  Submission rows: {len(submission)} | Mean probability: {submission[TARGET_COL].mean():.4f}")
    print(submission.head())

    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(f"  Baseline RF Score:             ~549.00")
    print(f"  LightGBM OOF Score:            {lgb_cv:.2f}")
    print(f"  CatBoost OOF Score:            {cat_cv:.2f}")
    print(f"  XGBoost OOF Score:             {xgb_cv:.2f}")
    print(f"  Ensemble (Optimal Weights):    {s_blend:.2f}")
    print(f"  Ensemble + Platt Calibration:  {s_cal:.2f}")
    print(f"  Total Time Elapsed:            {time.time()-t0:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
