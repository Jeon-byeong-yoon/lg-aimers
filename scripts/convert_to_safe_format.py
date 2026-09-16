"""
pickle 파일들을 버전 독립적 형식으로 변환
  pitcher_agg.pkl   → pitcher_agg.csv
  feature_config.pkl → feature_config.json + le_mappings.json
  iso_calibrator.pkl → iso_x.npy + iso_y.npy
"""
import pickle, json, os
import numpy as np
import pandas as pd

MODEL_DIR = r"scripts\submission_v2\model"

# ── 1. pitcher_agg: pickle → CSV ──────────────────────────────
print("1) pitcher_agg.pkl → pitcher_agg.csv")
with open(os.path.join(MODEL_DIR, "pitcher_agg.pkl"), "rb") as f:
    pitcher_agg = pickle.load(f)
csv_path = os.path.join(MODEL_DIR, "pitcher_agg.csv")
pitcher_agg.to_csv(csv_path, index=False, encoding="utf-8")
print(f"   저장: {csv_path}  ({os.path.getsize(csv_path)//1024}KB)")

# ── 2. feature_config: pickle → JSON + le_mappings JSON ───────
print("2) feature_config.pkl → feature_config.json + le_mappings.json")
with open(os.path.join(MODEL_DIR, "feature_config.pkl"), "rb") as f:
    cfg = pickle.load(f)

# median_vals는 dict로
median_dict = {k: float(v) for k, v in cfg["median_vals"].items()}

# LabelEncoder → plain dict {col: {str_val: int_idx}}
le_mappings = {}
for col, le in cfg["le_dict"].items():
    le_mappings[col] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}

feature_config_json = {
    "features":  cfg["features"],
    "cat_cols":  cfg["cat_cols"],
    "median_vals": median_dict,
}

json_path = os.path.join(MODEL_DIR, "feature_config.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(feature_config_json, f, ensure_ascii=False)
print(f"   저장: {json_path}")

le_path = os.path.join(MODEL_DIR, "le_mappings.json")
with open(le_path, "w", encoding="utf-8") as f:
    json.dump(le_mappings, f, ensure_ascii=False)
print(f"   저장: {le_path}")

# ── 3. iso_calibrator: pickle → npy 배열 2개 ──────────────────
print("3) iso_calibrator.pkl → iso_x.npy + iso_y.npy")
with open(os.path.join(MODEL_DIR, "iso_calibrator.pkl"), "rb") as f:
    iso = pickle.load(f)

np.save(os.path.join(MODEL_DIR, "iso_x.npy"), iso.X_thresholds_)
np.save(os.path.join(MODEL_DIR, "iso_y.npy"), iso.y_thresholds_)
print(f"   저장: iso_x.npy ({len(iso.X_thresholds_)}개 임계값)")
print(f"   저장: iso_y.npy")

print("\n변환 완료!")
print("이제 불필요한 pickle 파일들:")
for f in ["pitcher_agg.pkl", "feature_config.pkl", "iso_calibrator.pkl"]:
    p = os.path.join(MODEL_DIR, f)
    if os.path.exists(p):
        print(f"  삭제: {f}")
        os.remove(p)
