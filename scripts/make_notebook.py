import json

def code_cell(lines, cell_id=''):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {'id': cell_id},
        'outputs': [],
        'source': lines if isinstance(lines, list) else [lines]
    }

def md_cell(text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [text]
    }

cells = []

# ─── 타이틀 마크다운 ───────────────────────────────────────────
cells.append(md_cell(
    "# LG Aimers 9기 — 투구 제구 성공 확률 예측\n"
    "## 전략 1~5 통합 파이프라인 (Google Colab L4 GPU)\n\n"
    "| 전략 | 내용 |\n"
    "|------|------|\n"
    "| 1 | 확률 보정 (Isotonic Calibration) |\n"
    "| 2 | LightGBM + CatBoost + XGBoost GPU 학습 |\n"
    "| 3 | Trackman 피처 엔지니어링 |\n"
    "| 4 | 상성·폼 트렌드·압박 지표 파생 변수 |\n"
    "| 5 | 최적 가중치 앙상블 |\n\n"
    "**목표 점수**: 1,100+ (베이스라인 549점)"
))

# ─── 셀 1: 패키지 설치 ───────────────────────────────────────
cells.append(md_cell("## 셀 1: 패키지 설치"))
cells.append(code_cell(
    "!pip install catboost lightgbm xgboost scikit-learn pandas numpy scipy -q\n",
    'install'
))

# ─── 셀 2: 드라이브 마운트 ────────────────────────────────────
cells.append(md_cell(
    "## 셀 2: Google Drive 마운트\n"
    "> **★ 드라이브에 `lg_aimers/data/` 폴더를 만들고 4개 CSV를 업로드하세요**\n"
    "> - train.csv\n> - test.csv\n> - trackman_history.csv\n> - sample_submission.csv"
))
cells.append(code_cell([
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# ★ 아래 경로를 본인 드라이브 실제 경로로 수정하세요\n",
    "DATA_DIR = '/content/drive/MyDrive/lg_aimers/data'\n",
    "\n",
    "import os\n",
    "print('데이터 파일 확인:')\n",
    "for f in ['train.csv', 'test.csv', 'trackman_history.csv', 'sample_submission.csv']:\n",
    "    fp = os.path.join(DATA_DIR, f)\n",
    "    exists = os.path.exists(fp)\n",
    "    size = os.path.getsize(fp) // 1024 // 1024 if exists else 0\n",
    "    status = 'OK' if exists else 'MISSING'\n",
    "    print(f'  {f}: [{status}] {size}MB')\n",
], 'mount'))

# ─── 셀 3: 라이브러리 & 설정 ──────────────────────────────────
cells.append(md_cell("## 셀 3: 라이브러리 & 설정"))
cells.append(code_cell([
    "import os, gc, time, warnings, zipfile\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from scipy.optimize import minimize\n",
    "from sklearn.isotonic import IsotonicRegression\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from catboost import CatBoostClassifier, Pool\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# 경로 설정\n",
    "TRAIN_PATH   = os.path.join(DATA_DIR, 'train.csv')\n",
    "TEST_PATH    = os.path.join(DATA_DIR, 'test.csv')\n",
    "HISTORY_PATH = os.path.join(DATA_DIR, 'trackman_history.csv')\n",
    "SAMPLE_PATH  = os.path.join(DATA_DIR, 'sample_submission.csv')\n",
    "OUTPUT_DIR   = Path('/content/output')\n",
    "OUTPUT_DIR.mkdir(exist_ok=True)\n",
    "\n",
    "ID, TARGET     = 'row_id', 'control_success'\n",
    "CAT_COLS       = ['top_bottom', 'game_type', 'base_state']\n",
    "SEED, N_SPLITS = 42, 5\n",
    "print('설정 완료')\n",
], 'setup'))

# ─── 셀 4: 데이터 로드 ────────────────────────────────────────
cells.append(md_cell("## 셀 4: 데이터 로드"))
cells.append(code_cell([
    "print('데이터 로딩 중...')\n",
    "t0 = time.time()\n",
    "train = pd.read_csv(TRAIN_PATH, encoding='utf-8-sig')\n",
    "test  = pd.read_csv(TEST_PATH,  encoding='utf-8-sig')\n",
    "print(f'  train: {train.shape}  |  test: {test.shape}')\n",
    "print(f'  제구 성공률: {train[TARGET].mean():.4f}')\n",
    "print(f'  시즌: {train[\"season\"].min()} ~ {train[\"season\"].max()}')\n",
    "print(f'  로드 완료 ({time.time()-t0:.1f}s)')\n",
    "\n",
    "# 투수 ID 컬럼명 자동 탐지\n",
    "PITCHER_ID_COL = None\n",
    "for col in train.columns:\n",
    "    cl = col.lower()\n",
    "    if 'pitcher' in cl and ('id' in cl or 'trackman' in cl):\n",
    "        PITCHER_ID_COL = col\n",
    "        break\n",
    "print(f'  투수 ID 컬럼: {PITCHER_ID_COL}')\n",
    "print(f'  전체 컬럼: {list(train.columns)}')\n",
], 'load'))

# ─── 셀 5: Trackman FE ───────────────────────────────────────
cells.append(md_cell(
    "## 셀 5: Trackman 피처 엔지니어링 (전략 3)\n"
    "> 투수별 물리 지표 집계 → train/test에 join"
))
cells.append(code_cell([
    "print('Trackman 피처 엔지니어링 중...')\n",
    "t1 = time.time()\n",
    "\n",
    "TM_COLS = ['pitcher_trackman_id','season','pitch_type_group',\n",
    "           'rel_speed','spin_rate','induced_vert_break','horz_break',\n",
    "           'extension','rel_height','rel_side','zone_speed']\n",
    "history = pd.read_csv(HISTORY_PATH, encoding='utf-8-sig', usecols=TM_COLS)\n",
    "history = history[history['season'] <= 2024].copy()\n",
    "print(f'  trackman_history 로드: {history.shape} ({time.time()-t1:.1f}s)')\n",
    "\n",
    "# [A] 투수별 전체 누적 물리 집계\n",
    "pitcher_agg = history.groupby('pitcher_trackman_id').agg(\n",
    "    tm_speed_mean      =('rel_speed',          'mean'),\n",
    "    tm_speed_std       =('rel_speed',          'std'),\n",
    "    tm_spin_mean       =('spin_rate',          'mean'),\n",
    "    tm_spin_std        =('spin_rate',          'std'),\n",
    "    tm_vert_break_mean =('induced_vert_break', 'mean'),\n",
    "    tm_vert_break_std  =('induced_vert_break', 'std'),\n",
    "    tm_horz_break_mean =('horz_break',         'mean'),\n",
    "    tm_horz_break_std  =('horz_break',         'std'),\n",
    "    tm_rel_height_std  =('rel_height',         'std'),\n",
    "    tm_rel_side_std    =('rel_side',           'std'),\n",
    "    tm_extension_mean  =('extension',          'mean'),\n",
    "    tm_zone_speed_mean =('zone_speed',         'mean'),\n",
    "    tm_pitch_count     =('rel_speed',          'count'),\n",
    ").reset_index()\n",
    "pitcher_agg['tm_speed_drop'] = pitcher_agg['tm_speed_mean'] - pitcher_agg['tm_zone_speed_mean']\n",
    "pitcher_agg.drop(columns=['tm_zone_speed_mean'], inplace=True)\n",
    "\n",
    "# [B] 구종별 집계\n",
    "for grp_name in ['fastball', 'breaking', 'offspeed']:\n",
    "    sub = history[history['pitch_type_group'] == grp_name]\n",
    "    pt_agg = sub.groupby('pitcher_trackman_id').agg(\n",
    "        **{f'tm_{grp_name}_speed_mean': ('rel_speed',  'mean')},\n",
    "        **{f'tm_{grp_name}_spin_mean':  ('spin_rate',  'mean')},\n",
    "        **{f'tm_{grp_name}_rel_h_std':  ('rel_height', 'std')},\n",
    "        **{f'tm_{grp_name}_rel_s_std':  ('rel_side',   'std')},\n",
    "    ).reset_index()\n",
    "    pitcher_agg = pitcher_agg.merge(pt_agg, on='pitcher_trackman_id', how='left')\n",
    "\n",
    "# [C] 최근 1시즌(2024) 집계\n",
    "recent = history[history['season'] == 2024]\n",
    "recent_agg = recent.groupby('pitcher_trackman_id').agg(\n",
    "    tm_recent_speed_mean=('rel_speed',  'mean'),\n",
    "    tm_recent_spin_mean =('spin_rate',  'mean'),\n",
    "    tm_recent_rel_h_std =('rel_height', 'std'),\n",
    "    tm_recent_rel_s_std =('rel_side',   'std'),\n",
    ").reset_index()\n",
    "pitcher_agg = pitcher_agg.merge(recent_agg, on='pitcher_trackman_id', how='left')\n",
    "\n",
    "# [D] 컨디션 트렌드\n",
    "pitcher_agg['tm_speed_trend']     = pitcher_agg['tm_recent_speed_mean'] - pitcher_agg['tm_speed_mean']\n",
    "pitcher_agg['tm_release_trend_h'] = pitcher_agg['tm_recent_rel_h_std']  - pitcher_agg['tm_rel_height_std']\n",
    "pitcher_agg['tm_release_trend_s'] = pitcher_agg['tm_recent_rel_s_std']  - pitcher_agg['tm_rel_side_std']\n",
    "\n",
    "del history, recent, recent_agg\n",
    "gc.collect()\n",
    "print(f'  Trackman 피처 수: {pitcher_agg.shape[1]-1}개')\n",
    "\n",
    "# [E] train/test에 조인\n",
    "if PITCHER_ID_COL:\n",
    "    train = train.merge(pitcher_agg, left_on=PITCHER_ID_COL, right_on='pitcher_trackman_id', how='left')\n",
    "    test  = test.merge( pitcher_agg, left_on=PITCHER_ID_COL, right_on='pitcher_trackman_id', how='left')\n",
    "    for df in [train, test]:\n",
    "        df.drop(columns=['pitcher_trackman_id'], errors='ignore', inplace=True)\n",
    "    print(f'Trackman 조인 완료: train {train.shape}, test {test.shape}')\n",
    "else:\n",
    "    print('투수 ID 컬럼 탐지 실패 -> 위 셀 4 출력에서 올바른 컬럼명을 확인하세요')\n",
    "    print('예시: PITCHER_ID_COL = \"pitcher_id\"  # 이 줄을 수정해서 실행')\n",
], 'trackman_fe'))

# ─── 셀 6: 파생 변수 ─────────────────────────────────────────
cells.append(md_cell(
    "## 셀 6: 파생 변수 생성 (전략 4)\n"
    "> 최근 폼 트렌드, 투타 상성, 카운트·주자 압박, 릴리스 불안정성"
))
cells.append(code_cell([
    "def add_derived_features(df):\n",
    "    df = df.copy()\n",
    "    base  = 'asof_pitcher_success_rate'\n",
    "    prev1 = 'asof_pitcher_prev1_game_success_rate'\n",
    "    prev3 = 'asof_pitcher_prev3_game_success_rate'\n",
    "    prev5 = 'asof_pitcher_prev5_game_success_rate'\n",
    "\n",
    "    # 폼 트렌드 (최근 - 누적) 양수=상승세, 음수=슬럼프\n",
    "    for prev_col, label in [(prev1,'1'),(prev3,'3'),(prev5,'5')]:\n",
    "        if all(c in df.columns for c in [prev_col, base]):\n",
    "            df[f'feat_form_trend_{label}'] = df[prev_col] - df[base]\n",
    "\n",
    "    # 가중 폼 (최근일수록 가중치 높음)\n",
    "    form_map = {prev1:0.5, prev3:0.3, prev5:0.2}\n",
    "    avail = {k:v for k,v in form_map.items() if k in df.columns}\n",
    "    if avail and base in df.columns:\n",
    "        total_w = sum(avail.values())\n",
    "        df['feat_weighted_form'] = sum(\n",
    "            df[col].fillna(df[base].fillna(0.5)) * w for col, w in avail.items()\n",
    "        ) / total_w\n",
    "\n",
    "    # 투타 상성\n",
    "    if 'pitcher_hand' in df.columns and 'batter_hand' in df.columns:\n",
    "        matchup_map = {'Right_vs_Right':0,'Right_vs_Left':1,'Left_vs_Right':2,'Left_vs_Left':3}\n",
    "        df['feat_matchup'] = (\n",
    "            df['pitcher_hand'].astype(str) + '_vs_' + df['batter_hand'].astype(str)\n",
    "        ).map(matchup_map).fillna(-1).astype(int)\n",
    "\n",
    "    # 투수-타자 실력 격차\n",
    "    if 'asof_pitcher_success_rate' in df.columns and 'asof_batter_success_rate' in df.columns:\n",
    "        df['feat_pitcher_batter_edge'] = df['asof_pitcher_success_rate'] - df['asof_batter_success_rate']\n",
    "\n",
    "    # 카운트 압박\n",
    "    if all(c in df.columns for c in ['balls_before','strikes_before']):\n",
    "        df['feat_count_pressure'] = df['balls_before'] - df['strikes_before']\n",
    "        df['feat_full_count']     = ((df['balls_before']==3) & (df['strikes_before']==2)).astype(int)\n",
    "        df['feat_ahead_count']    = (df['strikes_before'] > df['balls_before']).astype(int)\n",
    "\n",
    "    # 주자 압박\n",
    "    runners = [c for c in ['runner_on_1b','runner_on_2b','runner_on_3b'] if c in df.columns]\n",
    "    if runners:\n",
    "        df['feat_runners_total'] = df[runners].sum(axis=1)\n",
    "        scoring = [c for c in ['runner_on_2b','runner_on_3b'] if c in df.columns]\n",
    "        if scoring:\n",
    "            df['feat_scoring_position'] = df[scoring].max(axis=1)\n",
    "\n",
    "    # 이닝 지표\n",
    "    if 'inning' in df.columns:\n",
    "        df['feat_late_inning']  = (df['inning'] >= 7).astype(int)\n",
    "        df['feat_extra_inning'] = (df['inning'] >= 10).astype(int)\n",
    "\n",
    "    # 볼/스트 비율 차이\n",
    "    if all(c in df.columns for c in ['asof_pitcher_strike_rate','asof_pitcher_ball_rate']):\n",
    "        df['feat_strike_ball_diff'] = df['asof_pitcher_strike_rate'] - df['asof_pitcher_ball_rate']\n",
    "\n",
    "    # 릴리스 포인트 불안정성\n",
    "    if 'tm_rel_height_std' in df.columns and 'tm_rel_side_std' in df.columns:\n",
    "        df['feat_release_instability'] = df['tm_rel_height_std'].fillna(0) + df['tm_rel_side_std'].fillna(0)\n",
    "\n",
    "    return df\n",
    "\n",
    "train = add_derived_features(train)\n",
    "test  = add_derived_features(test)\n",
    "print(f'파생 변수 생성 완료: train {train.shape}')\n",
], 'derived_fe'))

# ─── 셀 7: 피처 확정 & 전처리 ──────────────────────────────────
cells.append(md_cell("## 셀 7: 학습 피처 확정 & 전처리"))
cells.append(code_cell([
    "EXCLUDE = {ID, TARGET, 'pitcher_trackman_id', 'batter_trackman_id'}\n",
    "test_base_cols = set(pd.read_csv(TEST_PATH, encoding='utf-8-sig', nrows=0).columns) - {ID}\n",
    "derived_cols   = {c for c in train.columns if c.startswith('feat_') or c.startswith('tm_')}\n",
    "\n",
    "FEATURES = [\n",
    "    c for c in train.columns\n",
    "    if (c in test_base_cols or c in derived_cols) and c not in EXCLUDE\n",
    "]\n",
    "ALL_CAT_COLS = [c for c in CAT_COLS if c in FEATURES]\n",
    "print(f'총 피처: {len(FEATURES)}개 | 범주형: {ALL_CAT_COLS}')\n",
    "print(f'파생 피처: {len([c for c in FEATURES if c.startswith(\"feat_\") or c.startswith(\"tm_\")])}개')\n",
    "\n",
    "# LabelEncoding\n",
    "le_dict = {}\n",
    "for col in ALL_CAT_COLS:\n",
    "    le = LabelEncoder()\n",
    "    combined = pd.concat([train[col], test[col]], axis=0).astype(str)\n",
    "    le.fit(combined)\n",
    "    train[col] = le.transform(train[col].astype(str))\n",
    "    test[col]  = le.transform(test[col].astype(str))\n",
    "    le_dict[col] = le\n",
    "\n",
    "X      = train[FEATURES].copy()\n",
    "y      = train[TARGET].copy()\n",
    "X_test = test[FEATURES].copy()\n",
    "\n",
    "num_cols    = [c for c in FEATURES if c not in ALL_CAT_COLS]\n",
    "median_vals = X[num_cols].median()\n",
    "X[num_cols]      = X[num_cols].fillna(median_vals)\n",
    "X_test[num_cols] = X_test[num_cols].fillna(median_vals)\n",
    "\n",
    "print(f'X: {X.shape}  |  X_test: {X_test.shape}')\n",
    "\n",
    "# Brier Skill Score 함수\n",
    "def brier_skill_score(y_true, y_pred):\n",
    "    y_true = np.array(y_true)\n",
    "    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)\n",
    "    r = y_true.mean()\n",
    "    brier = ((y_pred - y_true) ** 2).mean()\n",
    "    base_brier = r * (1 - r)\n",
    "    return max(0.0, 100_000 * (1 - brier / base_brier)), brier, base_brier\n",
], 'feature_prep'))

# ─── 셀 8: LightGBM ─────────────────────────────────────────
cells.append(md_cell(
    "## 셀 8: LightGBM 학습 (전략 2) — L4 GPU\n"
    "> `'device': 'gpu'` 설정으로 L4 GPU를 자동 사용합니다"
))
cells.append(code_cell([
    "print('LightGBM 학습 시작...')\n",
    "lgb_params = {\n",
    "    'objective': 'binary', 'metric': 'binary_logloss',\n",
    "    'learning_rate': 0.05, 'num_leaves': 255, 'max_depth': -1,\n",
    "    'min_child_samples': 100, 'feature_fraction': 0.8,\n",
    "    'bagging_fraction': 0.8, 'bagging_freq': 5,\n",
    "    'lambda_l1': 0.1, 'lambda_l2': 0.1,\n",
    "    'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,\n",
    "    'n_jobs': -1, 'seed': SEED, 'verbose': -1,\n",
    "}\n",
    "\n",
    "kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)\n",
    "lgb_oof  = np.zeros(len(X))\n",
    "lgb_test = np.zeros(len(X_test))\n",
    "lgb_scores = []\n",
    "\n",
    "for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):\n",
    "    t_f = time.time()\n",
    "    X_tr, X_vl = X.iloc[trn_idx], X.iloc[val_idx]\n",
    "    y_tr, y_vl = y.iloc[trn_idx], y.iloc[val_idx]\n",
    "\n",
    "    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=ALL_CAT_COLS)\n",
    "    dval   = lgb.Dataset(X_vl, label=y_vl, categorical_feature=ALL_CAT_COLS, reference=dtrain)\n",
    "    m = lgb.train(\n",
    "        lgb_params, dtrain, num_boost_round=3000, valid_sets=[dval],\n",
    "        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],\n",
    "    )\n",
    "    lgb_oof[val_idx] = m.predict(X_vl)\n",
    "    lgb_test += m.predict(X_test) / N_SPLITS\n",
    "\n",
    "    s, b, _ = brier_skill_score(y_vl.values, lgb_oof[val_idx])\n",
    "    lgb_scores.append(s)\n",
    "    print(f'  Fold {fold+1} | Score: {s:.2f} | Brier: {b:.6f} | {time.time()-t_f:.1f}s')\n",
    "\n",
    "lgb_cv, lgb_brier, _ = brier_skill_score(y.values, lgb_oof)\n",
    "print(f'LightGBM CV Score: {lgb_cv:.2f}')\n",
], 'lgb'))

# ─── 셀 9: CatBoost ─────────────────────────────────────────
cells.append(md_cell(
    "## 셀 9: CatBoost 학습 (전략 2) — L4 GPU\n"
    "> `task_type='GPU'` 설정. 범주형 피처를 자동으로 처리합니다."
))
cells.append(code_cell([
    "print('CatBoost 학습 시작...')\n",
    "cat_params = dict(\n",
    "    iterations=3000, learning_rate=0.05, depth=8,\n",
    "    l2_leaf_reg=3, bagging_temperature=0.8, random_strength=1.0,\n",
    "    border_count=128, task_type='GPU', devices='0',\n",
    "    loss_function='Logloss', eval_metric='Logloss',\n",
    "    early_stopping_rounds=100, random_seed=SEED, verbose=500,\n",
    ")\n",
    "cat_oof  = np.zeros(len(X))\n",
    "cat_test = np.zeros(len(X_test))\n",
    "cat_feat_idx = [list(X.columns).index(c) for c in ALL_CAT_COLS if c in X.columns]\n",
    "\n",
    "for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):\n",
    "    t_f = time.time()\n",
    "    X_tr, X_vl = X.iloc[trn_idx], X.iloc[val_idx]\n",
    "    y_tr, y_vl = y.iloc[trn_idx], y.iloc[val_idx]\n",
    "    pool_tr = Pool(X_tr, label=y_tr, cat_features=cat_feat_idx)\n",
    "    pool_vl = Pool(X_vl, label=y_vl, cat_features=cat_feat_idx)\n",
    "    m = CatBoostClassifier(**cat_params)\n",
    "    m.fit(pool_tr, eval_set=pool_vl, use_best_model=True)\n",
    "    cat_oof[val_idx] = m.predict_proba(X_vl)[:, 1]\n",
    "    cat_test += m.predict_proba(X_test)[:, 1] / N_SPLITS\n",
    "    s, b, _ = brier_skill_score(y_vl.values, cat_oof[val_idx])\n",
    "    print(f'  Fold {fold+1} | Score: {s:.2f} | Brier: {b:.6f} | {time.time()-t_f:.1f}s')\n",
    "\n",
    "cat_cv, cat_brier, _ = brier_skill_score(y.values, cat_oof)\n",
    "print(f'CatBoost CV Score: {cat_cv:.2f}')\n",
], 'catboost'))

# ─── 셀 10: XGBoost ─────────────────────────────────────────
cells.append(md_cell(
    "## 셀 10: XGBoost 학습 (전략 2) — L4 GPU\n"
    "> `'device': 'cuda'` 설정"
))
cells.append(code_cell([
    "print('XGBoost 학습 시작...')\n",
    "xgb_params = {\n",
    "    'objective': 'binary:logistic', 'eval_metric': 'logloss',\n",
    "    'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 100,\n",
    "    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1,\n",
    "    'reg_alpha': 0.1, 'reg_lambda': 1.0,\n",
    "    'tree_method': 'hist', 'device': 'cuda',\n",
    "    'n_estimators': 3000, 'random_state': SEED, 'n_jobs': -1,\n",
    "}\n",
    "xgb_oof  = np.zeros(len(X))\n",
    "xgb_test = np.zeros(len(X_test))\n",
    "\n",
    "for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):\n",
    "    t_f = time.time()\n",
    "    X_tr, X_vl = X.iloc[trn_idx], X.iloc[val_idx]\n",
    "    y_tr, y_vl = y.iloc[trn_idx], y.iloc[val_idx]\n",
    "    m = xgb.XGBClassifier(**xgb_params)\n",
    "    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], early_stopping_rounds=100, verbose=500)\n",
    "    xgb_oof[val_idx] = m.predict_proba(X_vl)[:, 1]\n",
    "    xgb_test += m.predict_proba(X_test)[:, 1] / N_SPLITS\n",
    "    s, b, _ = brier_skill_score(y_vl.values, xgb_oof[val_idx])\n",
    "    print(f'  Fold {fold+1} | Score: {s:.2f} | Brier: {b:.6f} | {time.time()-t_f:.1f}s')\n",
    "\n",
    "xgb_cv, xgb_brier, _ = brier_skill_score(y.values, xgb_oof)\n",
    "print(f'XGBoost CV Score: {xgb_cv:.2f}')\n",
], 'xgboost'))

# ─── 셀 11: 앙상블 + 보정 ─────────────────────────────────────
cells.append(md_cell(
    "## 셀 11: 앙상블 + Isotonic Calibration (전략 1 & 5)\n"
    "> OOF 기반으로 최적 가중치를 Nelder-Mead로 탐색 후 Isotonic 보정 적용"
))
cells.append(code_cell([
    "print('앙상블 가중치 최적화 중...')\n",
    "\n",
    "def neg_bss(weights):\n",
    "    w = np.clip(weights, 0, 1)\n",
    "    w = w / w.sum()\n",
    "    blend = w[0]*lgb_oof + w[1]*cat_oof + w[2]*xgb_oof\n",
    "    s, _, _ = brier_skill_score(y.values, blend)\n",
    "    return -s\n",
    "\n",
    "result = minimize(neg_bss, [1/3, 1/3, 1/3], method='Nelder-Mead',\n",
    "                  options={'maxiter': 1000, 'xatol': 1e-7})\n",
    "opt_w = np.clip(result.x, 0, 1)\n",
    "opt_w /= opt_w.sum()\n",
    "print(f'  최적 가중치: LGB={opt_w[0]:.3f} | CAT={opt_w[1]:.3f} | XGB={opt_w[2]:.3f}')\n",
    "\n",
    "oof_blend  = opt_w[0]*lgb_oof  + opt_w[1]*cat_oof  + opt_w[2]*xgb_oof\n",
    "test_blend = opt_w[0]*lgb_test + opt_w[1]*cat_test + opt_w[2]*xgb_test\n",
    "\n",
    "s_blend, b_blend, _ = brier_skill_score(y.values, oof_blend)\n",
    "print(f'  앙상블 OOF Score: {s_blend:.2f} | Brier: {b_blend:.6f}')\n",
    "\n",
    "# Isotonic Calibration (전략 1)\n",
    "iso = IsotonicRegression(out_of_bounds='clip')\n",
    "iso.fit(oof_blend, y.values)\n",
    "oof_cal  = iso.predict(oof_blend)\n",
    "test_cal = np.clip(iso.predict(test_blend), 0.0, 1.0)\n",
    "\n",
    "s_cal, b_cal, _ = brier_skill_score(y.values, oof_cal)\n",
    "print(f'  보정 후 OOF Score: {s_cal:.2f} | Brier: {b_cal:.6f}')\n",
    "\n",
    "print()\n",
    "print('=' * 60)\n",
    "print('최종 비교')\n",
    "print('=' * 60)\n",
    "print(f'  베이스라인 RandomForest:  ~549')\n",
    "print(f'  LightGBM (단독):          {lgb_cv:.2f}')\n",
    "print(f'  CatBoost (단독):          {cat_cv:.2f}')\n",
    "print(f'  XGBoost  (단독):          {xgb_cv:.2f}')\n",
    "print(f'  앙상블 (최적 가중치):     {s_blend:.2f}')\n",
    "print(f'  앙상블 + 보정 (최종):     {s_cal:.2f}  <- 최종 제출')\n",
    "print('=' * 60)\n",
], 'ensemble'))

# ─── 셀 12: 제출 ─────────────────────────────────────────────
cells.append(md_cell("## 셀 12: 제출 파일 생성 & 다운로드"))
cells.append(code_cell([
    "sample_sub = pd.read_csv(SAMPLE_PATH, encoding='utf-8-sig')\n",
    "submission = pd.DataFrame({ID: test[ID].values, TARGET: test_cal})\n",
    "\n",
    "# 검증\n",
    "assert len(submission) == len(sample_sub), f'행 수 불일치: {len(submission)} vs {len(sample_sub)}'\n",
    "assert submission[TARGET].between(0, 1).all(), '확률 범위 초과'\n",
    "\n",
    "csv_path = OUTPUT_DIR / 'submission.csv'\n",
    "zip_path = OUTPUT_DIR / 'submission_full_pipeline.zip'\n",
    "\n",
    "submission.to_csv(csv_path, index=False)\n",
    "with zipfile.ZipFile(zip_path, 'w') as zf:\n",
    "    zf.write(csv_path, 'output/submission.csv')\n",
    "\n",
    "print(f'CSV 저장: {csv_path}')\n",
    "print(f'ZIP 저장: {zip_path}')\n",
    "print(f'  행 수: {len(submission):,}')\n",
    "print(f'  평균 확률: {submission[TARGET].mean():.4f}')\n",
    "print(f'  범위: [{submission[TARGET].min():.4f}, {submission[TARGET].max():.4f}]')\n",
    "print(f'OOF 점수: {s_cal:.2f}')\n",
    "\n",
    "# Colab에서 파일 다운로드\n",
    "from google.colab import files\n",
    "files.download(str(zip_path))\n",
    "print('다운로드 완료!')\n",
], 'submit'))

# ─── 노트북 구성 ────────────────────────────────────────────
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'accelerator': 'GPU',
        'colab': {'provenance': [], 'gpuType': 'L4'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.0'},
    },
    'cells': cells,
}

out_path = 'scripts/LG_Aimers_FullPipeline_Colab.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'노트북 생성 완료: {out_path}')
print(f'셀 수: {len(cells)}개')
