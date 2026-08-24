"""Freeze the end-of-2024 career as-of state used by V92 inference.

The anchor is built from official training rows only and is fixed at packaging
time. At inference each evaluation row is transformed using just its own
official ``asof_*`` values and this frozen table, so rows never interact.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, "scripts")
from inseason_asof_features_v92 import build_anchors, build_priors


OUTPUT = Path("artifacts/v92_inseason_anchor.joblib")
GROUPS = ("pitcher", "batter")
WEIGHT = 0.20


def main():
    columns = [
        "season", "pitcher_id", "batter_id",
        "asof_pitcher_n", "asof_batter_n",
        "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_strike_rate",
        "asof_batter_success_rate", "asof_batter_middle_rate",
    ]
    frame = pd.read_csv(
        "공모전 dataset/open/data/train.csv", encoding="utf-8-sig", usecols=columns
    )
    if int(frame["season"].max()) != 2024:
        raise ValueError("Anchor must be frozen at the end of the 2024 season")
    bundle = {
        "anchors": build_anchors(frame, GROUPS),
        "priors": build_priors(frame, GROUPS),
        "groups": list(GROUPS),
        "weight": WEIGHT,
        "target_season": 2025,
        "anchor_season_max": int(frame["season"].max()),
        "train_rows": int(len(frame)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT, compress=3)
    sizes = {name: int(len(table)) for name, table in bundle["anchors"].items()}
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size / 2**10:.1f} KiB)")
    print(f"anchor rows={sizes} weight={WEIGHT} priors={bundle['priors']}")


if __name__ == "__main__":
    main()
