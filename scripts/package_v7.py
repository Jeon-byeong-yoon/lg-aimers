"""
Package submission_v7_final.zip for DACON submission.
Includes:
  - script.py (LGBM + CatBoost)
  - requirements.txt (lightgbm, catboost)
  - model/ (native model files, JSON configs, Isotonic params, weights)
  - output/submission.csv
"""
import os
import zipfile
from pathlib import Path

BASE_DIR = Path(r"c:\Users\wnsgu\Desktop\lg-aimers")
SUBMIT_DIR = BASE_DIR / "scripts" / "submission_v2"
ZIP_OUT_PATH = BASE_DIR / "scripts" / "submission_v7_final.zip"

req_text = """catboost>=1.2.0
lightgbm>=4.0.0
"""
with open(SUBMIT_DIR / "requirements.txt", "w", encoding="utf-8") as f:
    f.write(req_text)

print("Packaging submission_v7_final.zip...")
with zipfile.ZipFile(ZIP_OUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(SUBMIT_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, SUBMIT_DIR)
            zf.write(fpath, arcname)

size_mb = os.path.getsize(ZIP_OUT_PATH) / 1024 / 1024
print(f"ZIP created: {ZIP_OUT_PATH} ({size_mb:.2f} MB)")
print("ZIP Contents:")
with zipfile.ZipFile(ZIP_OUT_PATH, "r") as zf:
    for name in sorted(zf.namelist()):
        kb = zf.getinfo(name).file_size / 1024
        print(f"  {name} ({kb:.1f} KB)")
