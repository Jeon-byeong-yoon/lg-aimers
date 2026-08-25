"""Run a packaged submission the way the evaluation server does, then prove row independence.

Two things are checked, both of which have bitten this project before.

Timing: the server allows ten minutes for model loading, preprocessing, prediction
and CSV writing combined. Local hardware is not the server's, but a run that is
already close to the limit here is a warning.

Row independence: the rules forbid a prediction for one evaluation row from
depending on any other evaluation row. Every submission is therefore run twice, once
on the full test file and once on a random subset, and the shared rows must receive
identical values. Any leak through a group statistic, a fitted transform or a
sort-order dependence shows up as a difference here.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

SERVER_PYTHON = "/opt/anaconda3/envs/lgaimers_server/bin/python"
OFFICIAL = Path("공모전 dataset/open/data")
SUBSET_ROWS = 20000
SEED = 20260825


def stage(archive, root, test_frame, submission_frame):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    with ZipFile(archive) as zf:
        zf.extractall(root)
    (root / "data").mkdir()
    test_frame.to_csv(root / "data/test.csv", index=False, encoding="utf-8-sig")
    submission_frame.to_csv(
        root / "data/sample_submission.csv", index=False, encoding="utf-8-sig")
    for name in ("trackman_history.csv", "train.csv"):
        source = OFFICIAL / name
        if source.exists():
            (root / "data" / name).symlink_to(source.resolve())
    return root


def run(root, label):
    started = time.time()
    result = subprocess.run([SERVER_PYTHON, "script.py"], cwd=root,
                            capture_output=True, text=True)
    elapsed = time.time() - started
    print(f"[{label}] exit={result.returncode} elapsed={elapsed:.0f}s")
    if result.stdout.strip():
        print("  " + result.stdout.strip().replace("\n", "\n  "))
    if result.returncode != 0:
        print("  " + result.stderr.strip()[-3000:].replace("\n", "\n  "))
        raise SystemExit(f"{label} run failed")
    output = pd.read_csv(root / "output/submission.csv")
    return output, elapsed


def main():
    archive = Path(sys.argv[1])
    sandbox = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/smoke")
    # The official `test.csv` shipped locally holds five rows, so timing and row
    # independence need a full-size stand-in. Pass its directory as the third
    # argument; the official files are still used for train and trackman history.
    data = Path(sys.argv[3]) if len(sys.argv) > 3 else OFFICIAL
    test = pd.read_csv(data / "test.csv", encoding="utf-8-sig")
    submission = pd.read_csv(data / "sample_submission.csv", encoding="utf-8-sig")
    print(f"{archive} ({archive.stat().st_size / 2**20:.1f} MiB), "
          f"test rows={len(test):,}")

    full, elapsed = run(stage(archive, sandbox / "full", test, submission), "full")
    if elapsed > 600:
        print("  WARNING: exceeded the server's ten-minute inference budget locally")

    if len(test) < 1000:
        print("test frame too small for a meaningful subset comparison; stopping here")
        print(f"prediction mean={full['control_success'].mean():.6f}")
        print("PASS (single run only)")
        return

    rng = np.random.default_rng(SEED)
    picked = rng.choice(len(test), size=min(SUBSET_ROWS, len(test)), replace=False)
    # Shuffled, not merely truncated: a sort-order dependence survives truncation.
    subset_test = test.iloc[picked].reset_index(drop=True)
    subset_submission = submission[
        submission["row_id"].isin(subset_test["row_id"])].reset_index(drop=True)
    subset, _ = run(
        stage(archive, sandbox / "subset", subset_test, subset_submission), "subset")

    merged = full.merge(subset, on="row_id", suffixes=("_full", "_subset"))
    difference = np.abs(merged["control_success_full"]
                        - merged["control_success_subset"]).max()
    print(f"\nrow independence: compared {len(merged):,} shared rows, "
          f"max abs difference {difference:.3e}")
    if difference > 1e-9:
        raise SystemExit("Row independence violated")
    print(f"prediction mean={full['control_success'].mean():.6f}  "
          f"min={full['control_success'].min():.6f}  "
          f"max={full['control_success'].max():.6f}")
    print("PASS")


if __name__ == "__main__":
    main()
