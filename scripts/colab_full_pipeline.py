"""
LG Aimers 9기 — 투구 제구 성공 확률 예측
==========================================
전략 1~5 통합 파이프라인 (Google Colab L4 GPU용)

실행 환경: Google Colab (L4 GPU)
목표 점수: 1,100+ (현재 549점 → 상위권 도전)

사용 전략:
  1. 확률 보정 (Probability Calibration)
  2. CatBoost / LightGBM / XGBoost 모델 도입
  3. Trackman 피처 엔지니어링 (제구 일관성, 구속, 피로도)
  4. 상성 & 최근 폼 파생 변수
  5. 앙상블 (CatBoost + LightGBM + XGBoost + Calibration)
"""

# ============================================================
# 셀 1: 패키지 설치 (Colab에서 맨 처음 실행)
# ============================================================
# 코랩에서 아래 주석 해제 후 실행
# !pip install catboost lightgbm xgboost scikit-learn pandas numpy scipy -q

# ============================================================
# 셀 2: 라이브러리 & 설정
# ============================================================
import os
import gc
import time
import warnings
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# ★ Google Drive 마운트 후 이 경로를 수정하세요
# ──────────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = Path("/content/drive/MyDrive/lg_aimers/data")  # ← 실제 경로로 변경

DATA_DIR     = Path("./공모전 dataset/open/data")   # 로컬 테스트용
TRAIN_PATH   = DATA_DIR / "train.csv"
TEST_PATH    = DATA_DIR / "test.csv"
HISTORY_PATH = DATA_DIR / "trackman_history.csv"
SAMPLE_PATH  = DATA_DIR / "sample_submission.csv"
OUTPUT_DIR   = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

ID       = "row_id"
TARGET   = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
SEED     = 42
N_SPLITS = 5

print("✅ 환경 설정 완료")

# ============================================================
# 셀 3: 데이터 로드
# ============================================================
print("📂 데이터 로딩 중...")
t0 = time.time()

train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
test  = pd.read_csv(TEST_PATH,  encoding="utf-8-sig")

print(f"  train: {train.shape}  |  test: {test.shape}")
print(f"  제구 성공률: {train[TARGET].mean():.4f}")
print(f"  시즌: {train['season'].min()} ~ {train['season'].max()}")
print(f"  로드 완료 ({time.time()-t0:.1f}s)")

# ============================================================
# 셀 4: 투수 ID 컬럼명 자동 탐지
# ============================================================
# train.csv에서 투수 ID 컬럼명을 자동 탐지합니다.
PITCHER_ID_COL = None
BATTER_ID_COL  = None
for col in train.columns:
    cl = col.lower()
    if PITCHER_ID_COL is None and "pitcher" in cl and ("id" in cl or "trackman" in cl):
        PITCHER_ID_COL = col
    if BATTER_ID_COL is None and "batter" in cl and ("id" in cl or "trackman" in cl):
        BATTER_ID_COL = col

print(f"  탐지된 투수 ID 컬럼: {PITCHER_ID_COL}")
print(f"  탐지된 타자 ID 컬럼: {BATTER_ID_COL}")

# ============================================================
# 셀 5: Trackman 피처 엔지니어링 (전략 3)
# ============================================================
print("\n📐 Trackman 피처 엔지니어링 중...")
t1 = time.time()

TM_COLS = [
    "pitcher_trackman_id", "season",
    "pitch_type_group",
    "rel_speed", "spin_rate",
    "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
]

history = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig", usecols=TM_COLS)
# 2025 시즌(test 미래 데이터) 제외
history = history[history["season"] <= 2024].copy()
print(f"  trackman_history 로드: {history.shape} ({time.time()-t1:.1f}s)")

# ── [A] 투수별 전체 누적 물리 집계 ──
pitcher_agg = history.groupby("pitcher_trackman_id").agg(
    tm_speed_mean        =("rel_speed",           "mean"),
    tm_speed_std         =("rel_speed",           "std"),
    tm_spin_mean         =("spin_rate",           "mean"),
    tm_spin_std          =("spin_rate",           "std"),
    tm_vert_break_mean   =("induced_vert_break",  "mean"),
    tm_vert_break_std    =("induced_vert_break",  "std"),
    tm_horz_break_mean   =("horz_break",          "mean"),
    tm_horz_break_std    =("horz_break",          "std"),
    tm_rel_height_std    =("rel_height",          "std"),
    tm_rel_side_std      =("rel_side",            "std"),
    tm_extension_mean    =("extension",           "mean"),
    tm_zone_speed_mean   =("zone_speed",          "mean"),
    tm_pitch_count       =("rel_speed",           "count"),
).reset_index()
tm_speed_drop = (
    history.groupby("pitcher_trackman_id")["rel_speed"].mean()
    - history.groupby("pitcher_trackman_id")["zone_speed"].mean()
)
pitcher_agg["tm_speed_drop"] = pitcher_agg["pitcher_trackman_id"].map(tm_speed_drop)

# ── [B] 구종별 집계 ──
for grp_name in ["fastball", "breaking", "offspeed"]:
    sub = history[history["pitch_type_group"] == grp_name]
    pt_agg = sub.groupby("pitcher_trackman_id").agg(
        **{f"tm_{grp_name}_speed_mean": ("rel_speed",   "mean")},
        **{f"tm_{grp_name}_spin_mean":  ("spin_rate",   "mean")},
        **{f"tm_{grp_name}_rel_h_std":  ("rel_height",  "std")},
        **{f"tm_{grp_name}_rel_s_std":  ("rel_side",    "std")},
    ).reset_index()
    pitcher_agg = pitcher_agg.merge(pt_agg, on="pitcher_trackman_id", how="left")

# ── [C] 최근 1시즌(2024) 집계 ──
recent = history[history["season"] == 2024]
recent_agg = recent.groupby("pitcher_trackman_id").agg(
    tm_recent_speed_mean =("rel_speed",  "mean"),
    tm_recent_spin_mean  =("spin_rate",  "mean"),
    tm_recent_rel_h_std  =("rel_height", "std"),
    tm_recent_rel_s_std  =("rel_side",   "std"),
).reset_index()
pitcher_agg = pitcher_agg.merge(recent_agg, on="pitcher_trackman_id", how="left")

# ── [D] 최근 컨디션 트렌드 ──
# 전체 평균 대비 최근 1시즌 구속 차이 (양수 = 최근 빠름, 음수 = 피로)
pitcher_agg["tm_speed_trend"] = pitcher_agg["tm_recent_speed_mean"] - pitcher_agg["tm_speed_mean"]
# 전체 대비 최근 릴리스 불안정성 변화
pitcher_agg["tm_release_trend_h"] = pitcher_agg["tm_recent_rel_h_std"] - pitcher_agg["tm_rel_height_std"]
pitcher_agg["tm_release_trend_s"] = pitcher_agg["tm_recent_rel_s_std"] - pitcher_agg["tm_rel_side_std"]

del history, recent, recent_agg
gc.collect()
print(f"  Trackman 피처 수: {pitcher_agg.shape[1]-1}개")

# ── [E] train/test에 조인 ──
if PITCHER_ID_COL:
    train = train.merge(pitcher_agg, left_on=PITCHER_ID_COL, right_on="pitcher_trackman_id", how="left")
    test  = test.merge(pitcher_agg,  left_on=PITCHER_ID_COL, right_on="pitcher_trackman_id", how="left")
    train.drop(columns=["pitcher_trackman_id"], errors="ignore", inplace=True)
    test.drop(columns=["pitcher_trackman_id"],  errors="ignore", inplace=True)
    print(f"✅ Trackman 조인 완료: train {train.shape}, test {test.shape}")
else:
    print("⚠️  투수 ID 컬럼 미탐지 → Trackman 조인 건너뜀")

# ============================================================
# 셀 6: 파생 변수 생성 (전략 4)
# ============================================================
print("\n🔧 파생 변수 생성 중...")

def add_derived_features(df, train_median=None):
    df = df.copy()

    # ── 최근 폼 트렌드 ──
    prev1 = "asof_pitcher_prev1_game_success_rate"
    prev3 = "asof_pitcher_prev3_game_success_rate"
    prev5 = "asof_pitcher_prev5_game_success_rate"
    base  = "asof_pitcher_success_rate"
    if all(c in df.columns for c in [prev1, base]):
        df["feat_form_trend_1"] = df[prev1] - df[base]
    if all(c in df.columns for c in [prev3, base]):
        df["feat_form_trend_3"] = df[prev3] - df[base]
    if all(c in df.columns for c in [prev5, base]):
        df["feat_form_trend_5"] = df[prev5] - df[base]

    # 가중 폼 지표 (최근일수록 가중치 높음)
    form_cols = {prev1: 0.5, prev3: 0.3, prev5: 0.2}
    avail = {k: v for k, v in form_cols.items() if k in df.columns}
    if avail:
        total_w = sum(avail.values())
        df["feat_weighted_form"] = sum(
            df[col].fillna(df[base].fillna(0.5)) * w for col, w in avail.items()
        ) / total_w

    # ── 투타 상성 ──
    if "pitcher_hand" in df.columns and "batter_hand" in df.columns:
        matchup_map = {"Right_vs_Right": 0, "Right_vs_Left": 1, "Left_vs_Right": 2, "Left_vs_Left": 3}
        df["feat_matchup"] = (df["pitcher_hand"].astype(str) + "_vs_" + df["batter_hand"].astype(str)).map(matchup_map).fillna(-1).astype(int)

    # ── 투수-타자 실력 격차 ──
    if "asof_pitcher_success_rate" in df.columns and "asof_batter_success_rate" in df.columns:
        df["feat_pitcher_batter_edge"] = df["asof_pitcher_success_rate"] - df["asof_batter_success_rate"]

    # ── 카운트 압박 ──
    if all(c in df.columns for c in ["balls_before", "strikes_before"]):
        df["feat_count_pressure"] = df["balls_before"] - df["strikes_before"]
        df["feat_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
        df["feat_ahead_count"] = (df["strikes_before"] > df["balls_before"]).astype(int)

    # ── 주자 압박 ──
    runner_cols = [c for c in ["runner_on_1b", "runner_on_2b", "runner_on_3b"] if c in df.columns]
    if runner_cols:
        df["feat_runners_total"] = df[runner_cols].sum(axis=1)
        scoring = [c for c in ["runner_on_2b", "runner_on_3b"] if c in df.columns]
        if scoring:
            df["feat_scoring_position"] = df[scoring].max(axis=1)

    # ── 이닝 지표 ──
    if "inning" in df.columns:
        df["feat_late_inning"]  = (df["inning"] >= 7).astype(int)
        df["feat_extra_inning"] = (df["inning"] >= 10).astype(int)

    # ── 볼/스트 비율 차이 ──
    if all(c in df.columns for c in ["asof_pitcher_strike_rate", "asof_pitcher_ball_rate"]):
        df["feat_strike_ball_diff"] = df["asof_pitcher_strike_rate"] - df["asof_pitcher_ball_rate"]

    # ── 릴리스 포인트 불안정성 ──
    if "tm_rel_height_std" in df.columns and "tm_rel_side_std" in df.columns:
        df["feat_release_instability"] = (
            df["tm_rel_height_std"].fillna(0) + df["tm_rel_side_std"].fillna(0)
        )

    # ── 구속 안정성 ──
    if "tm_speed_std" in df.columns:
        df["feat_speed_instability"] = df["tm_speed_std"].fillna(df["tm_speed_std"].median() if train_median is None else train_median.get("tm_speed_std", 0))

    return df

train = add_derived_features(train)
test  = add_derived_features(test)
print(f"✅ 파생 변수 생성 완료: train {train.shape}")

# ============================================================
# 셀 7: 학습 피처 확정 & 전처리
# ============================================================
print("\n⚙️ 피처 선택 및 전처리...")

EXCLUDE = {ID, TARGET, "pitcher_trackman_id", "batter_trackman_id"}
test_base_cols = set(pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns) - {ID}
derived_cols   = {c for c in train.columns if c.startswith("feat_") or c.startswith("tm_")}

FEATURES = [
    c for c in train.columns
    if (c in test_base_cols or c in derived_cols) and c not in EXCLUDE
]
print(f"  총 피처 수: {len(FEATURES)}개 (기본: {len(test_base_cols)} + 파생: {len(derived_cols & set(FEATURES))})")

ALL_CAT_COLS = [c for c in CAT_COLS if c in FEATURES]
print(f"  범주형: {ALL_CAT_COLS}")

# LabelEncoding (LightGBM/XGBoost용)
le_dict = {}
for col in ALL_CAT_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    le_dict[col] = le

X      = train[FEATURES].copy()
y      = train[TARGET].copy()
X_test = test[FEATURES].copy()

# 수치형 결측값 처리
num_cols = [c for c in FEATURES if c not in ALL_CAT_COLS]
median_vals = X[num_cols].median()
X[num_cols]      = X[num_cols].fillna(median_vals)
X_test[num_cols] = X_test[num_cols].fillna(median_vals)

print(f"✅ X: {X.shape}  |  X_test: {X_test.shape}")

# ============================================================
# 셀 8: Brier Skill Score 함수
# ============================================================
def brier_skill_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    r = y_true.mean()
    brier = ((y_pred - y_true) ** 2).mean()
    baseline_brier = r * (1 - r)
    score = max(0.0, 100_000 * (1 - brier / baseline_brier))
    return score, brier, baseline_brier

# ============================================================
# 셀 9: LightGBM 학습 (전략 2) — GPU 가속
# ============================================================
print("\n🚀 LightGBM 학습 시작...")

lgb_params = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "learning_rate":    0.05,
    "num_leaves":       255,
    "max_depth":        -1,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "device":           "gpu",   # Colab L4
    "gpu_platform_id":  0,
    "gpu_device_id":    0,
    "n_jobs":           -1,
    "seed":             SEED,
    "verbose":          -1,
}

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

lgb_oof  = np.zeros(len(X))
lgb_test = np.zeros(len(X_test))
lgb_scores = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    t_fold = time.time()
    X_tr, X_vl = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_vl = y.iloc[trn_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=ALL_CAT_COLS)
    dval   = lgb.Dataset(X_vl, label=y_vl, categorical_feature=ALL_CAT_COLS, reference=dtrain)

    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(500),
        ],
    )
    lgb_oof[val_idx] = model.predict(X_vl)
    lgb_test += model.predict(X_test) / N_SPLITS

    s, b, _ = brier_skill_score(y_vl.values, lgb_oof[val_idx])
    lgb_scores.append(s)
    print(f"  Fold {fold+1} | Score: {s:.2f} | Brier: {b:.6f} | {time.time()-t_fold:.1f}s")

lgb_cv, lgb_brier, _ = brier_skill_score(y.values, lgb_oof)
print(f"\n✅ LightGBM CV Score: {lgb_cv:.2f}")

# ============================================================
# 셀 10: CatBoost 학습 (전략 2) — GPU 가속
# ============================================================
print("\n🐱 CatBoost 학습 시작...")

cat_params = dict(
    iterations          = 3000,
    learning_rate       = 0.05,
    depth               = 8,
    l2_leaf_reg         = 3,
    bagging_temperature = 0.8,
    random_strength     = 1.0,
    border_count        = 128,
    task_type           = "GPU",
    devices             = "0",
    loss_function       = "Logloss",
    eval_metric         = "Logloss",
    early_stopping_rounds = 100,
    random_seed         = SEED,
    verbose             = 500,
)

cat_oof  = np.zeros(len(X))
cat_test = np.zeros(len(X_test))
cat_scores = []

cat_feat_idx = [list(X.columns).index(c) for c in ALL_CAT_COLS if c in X.columns]

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    t_fold = time.time()
    X_tr, X_vl = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_vl = y.iloc[trn_idx], y.iloc[val_idx]

    pool_tr = Pool(X_tr, label=y_tr, cat_features=cat_feat_idx)
    pool_vl = Pool(X_vl, label=y_vl, cat_features=cat_feat_idx)

    model = CatBoostClassifier(**cat_params)
    model.fit(pool_tr, eval_set=pool_vl, use_best_model=True)

    cat_oof[val_idx] = model.predict_proba(X_vl)[:, 1]
    cat_test += model.predict_proba(X_test)[:, 1] / N_SPLITS

    s, b, _ = brier_skill_score(y_vl.values, cat_oof[val_idx])
    cat_scores.append(s)
    print(f"  Fold {fold+1} | Score: {s:.2f} | Brier: {b:.6f} | {time.time()-t_fold:.1f}s")

cat_cv, cat_brier, _ = brier_skill_score(y.values, cat_oof)
print(f"\n✅ CatBoost CV Score: {cat_cv:.2f}")

# ============================================================
# 셀 11: XGBoost 학습 (전략 2) — GPU 가속
# ============================================================
print("\n⚡ XGBoost 학습 시작...")

xgb_params = {
    "objective":       "binary:logistic",
    "eval_metric":     "logloss",
    "learning_rate":   0.05,
    "max_depth":       8,
    "min_child_weight": 100,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "gamma":           0.1,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "tree_method":     "hist",
    "device":          "cuda",   # Colab L4
    "n_estimators":    3000,
    "random_state":    SEED,
    "n_jobs":          -1,
}

xgb_oof  = np.zeros(len(X))
xgb_test = np.zeros(len(X_test))
xgb_scores = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    t_fold = time.time()
    X_tr, X_vl = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_vl = y.iloc[trn_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        early_stopping_rounds=100,
        verbose=500,
    )
    xgb_oof[val_idx] = model.predict_proba(X_vl)[:, 1]
    xgb_test += model.predict_proba(X_test)[:, 1] / N_SPLITS

    s, b, _ = brier_skill_score(y_vl.values, xgb_oof[val_idx])
    xgb_scores.append(s)
    print(f"  Fold {fold+1} | Score: {s:.2f} | Brier: {b:.6f} | {time.time()-t_fold:.1f}s")

xgb_cv, xgb_brier, _ = brier_skill_score(y.values, xgb_oof)
print(f"\n✅ XGBoost CV Score: {xgb_cv:.2f}")

# ============================================================
# 셀 12: 최적 앙상블 가중치 탐색 (전략 5)
# ============================================================
print("\n🎯 앙상블 가중치 최적화 중...")

def neg_bss(weights):
    w = np.clip(weights, 0, 1)
    w = w / w.sum()
    blend = w[0]*lgb_oof + w[1]*cat_oof + w[2]*xgb_oof
    s, _, _ = brier_skill_score(y.values, blend)
    return -s

result = minimize(
    neg_bss, [1/3, 1/3, 1/3],
    method="Nelder-Mead",
    options={"maxiter": 1000, "xatol": 1e-7},
)
opt_w = np.clip(result.x, 0, 1)
opt_w /= opt_w.sum()
print(f"  최적 가중치: LGB={opt_w[0]:.3f} | CAT={opt_w[1]:.3f} | XGB={opt_w[2]:.3f}")

# OOF 앙상블
oof_blend  = opt_w[0]*lgb_oof  + opt_w[1]*cat_oof  + opt_w[2]*xgb_oof
test_blend = opt_w[0]*lgb_test + opt_w[1]*cat_test + opt_w[2]*xgb_test

blend_score, blend_brier, _ = brier_skill_score(y.values, oof_blend)
print(f"  앙상블 OOF Score: {blend_score:.2f} | Brier: {blend_brier:.6f}")

# ============================================================
# 셀 13: Isotonic Calibration (전략 1)
# ============================================================
print("\n📐 Isotonic Calibration 적용 중...")

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_blend, y.values)

oof_calibrated  = iso.predict(oof_blend)
test_calibrated = iso.predict(test_blend)
test_calibrated = np.clip(test_calibrated, 0.0, 1.0)

cal_score, cal_brier, _ = brier_skill_score(y.values, oof_calibrated)
print(f"  보정 후 OOF Score: {cal_score:.2f} | Brier: {cal_brier:.6f}")

# ============================================================
# 셀 14: 최종 점수 비교 요약
# ============================================================
print("\n" + "="*60)
print("📊 최종 모델별 OOF Brier Skill Score 비교")
print("="*60)
print(f"  베이스라인 RandomForest:  ~549")
print(f"  LightGBM (단독):          {lgb_cv:.2f}")
print(f"  CatBoost (단독):          {cat_cv:.2f}")
print(f"  XGBoost  (단독):          {xgb_cv:.2f}")
print(f"  앙상블 (최적 가중치):     {blend_score:.2f}")
print(f"  앙상블 + 보정 (최종):     {cal_score:.2f}  ★")
print("="*60)

# ============================================================
# 셀 15: 제출 파일 생성 & ZIP
# ============================================================
print("\n📝 제출 파일 생성...")

sample_sub = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
submission = pd.DataFrame({
    ID:     test[ID].values,
    TARGET: test_calibrated,
})

# 검증
assert len(submission) == len(sample_sub), f"행 수 불일치: {len(submission)} vs {len(sample_sub)}"
assert submission[TARGET].between(0, 1).all(), "확률 범위 초과!"

csv_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(csv_path, index=False)
print(f"✅ CSV 저장: {csv_path}")
print(f"   행 수: {len(submission):,}")
print(f"   평균 확률: {submission[TARGET].mean():.4f}")
print(f"   범위: [{submission[TARGET].min():.4f}, {submission[TARGET].max():.4f}]")

zip_path = OUTPUT_DIR / "submission_full_pipeline.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    zf.write(csv_path, "output/submission.csv")
print(f"✅ ZIP 저장: {zip_path}")
print(f"\n🏆 전체 파이프라인 완료! OOF 점수: {cal_score:.2f}")
