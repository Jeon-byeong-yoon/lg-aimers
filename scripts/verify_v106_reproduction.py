"""Check that a rebuilt submit_v106.zip predicts exactly what was submitted.

Reproduction is only meaningful if the rebuilt model scores the same. The recorded
reference predictions are the five distributed sample rows, whose values were
captured from the ZIP that produced Public 973.0643764994. All models use fixed
seeds and no GPU, so a faithful rebuild must match to floating-point noise.

Also re-checks the two properties the submission has to hold regardless of
rebuild: every artifact loads under the server's package versions, and each
evaluation row is predicted independently of the others.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "submissions/by_submitter/전병윤_RF_기준선/submit_v106.zip"
DATA = ROOT / "공모전 dataset/open/data"
TOLERANCE = 1e-9
# Captured from the ZIP that scored Public 973.0643764994.
REFERENCE = {
    "TEST_000001": 0.395789,
    "TEST_000017": 0.412843,
    "TEST_000213": 0.447886,
    "TEST_005332": 0.508071,
    "TEST_035185": 0.492471,
}


def run(workdir):
    result = subprocess.run(
        [sys.executable, "script.py"], cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise SystemExit("inference failed")
    return pd.read_csv(workdir / "output/submission.csv")


def main():
    if not ZIP.exists():
        raise SystemExit(f"missing {ZIP}")
    failures = []
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        with zipfile.ZipFile(ZIP) as archive:
            archive.extractall(work)
        (work / "data").mkdir(exist_ok=True)
        for name in ("test.csv", "sample_submission.csv"):
            shutil.copy(DATA / name, work / "data" / name)

        print("1. artifacts load under this interpreter")
        import joblib
        for path in sorted((work / "model").glob("*.joblib")):
            try:
                joblib.load(path)
                print(f"   OK   model/{path.name}")
            except Exception as error:
                failures.append(f"load {path.name}: {error}")
                print(f"   FAIL model/{path.name}: {error}")

        print("2. predictions match the recorded submission")
        batch = run(work).set_index("row_id")["control_success"]
        worst = 0.0
        for row_id, expected in REFERENCE.items():
            actual = float(batch.loc[row_id])
            difference = abs(actual - expected)
            worst = max(worst, difference)
            flag = "OK  " if difference <= 5e-6 else "FAIL"
            print(f"   {flag} {row_id}  expected {expected:.6f}  got {actual:.6f}"
                  f"  diff {difference:.2e}")
            if difference > 5e-6:
                failures.append(f"{row_id} differs by {difference:.2e}")
        print(f"   worst difference {worst:.2e} (reference stored to 6 decimals)")

        print("3. each evaluation row is predicted independently")
        test = pd.read_csv(work / "data/test.csv", encoding="utf-8-sig")
        sample = pd.read_csv(work / "data/sample_submission.csv", encoding="utf-8-sig")
        singles = []
        for position in range(len(test)):
            test.iloc[[position]].to_csv(
                work / "data/test.csv", index=False, encoding="utf-8-sig")
            row_id = test["row_id"].iloc[position]
            sample[sample["row_id"] == row_id].to_csv(
                work / "data/sample_submission.csv", index=False, encoding="utf-8-sig")
            singles.append(run(work))
        one_by_one = pd.concat(singles, ignore_index=True).set_index("row_id")
        gap = max(abs(float(one_by_one.loc[i, "control_success"]) - float(batch.loc[i]))
                  for i in batch.index)
        print(f"   max |batch - one-by-one| = {gap:.2e}")
        if gap > TOLERANCE:
            failures.append(f"row independence violated by {gap:.2e}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("reproduction verified: artifacts load, predictions match, rows independent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
