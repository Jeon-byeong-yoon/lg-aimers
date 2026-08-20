"""Build fixed V14 fine-tuned segment calibration from chronological OOF."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_v12_calibration import COMPONENTS, WEIGHTS, make_lookup


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    indices, targets, predictions = [], [], []
    for year in (2022, 2023, 2024):
        item = oof[str(year)]
        matrix = np.column_stack([item["components"][name] for name in COMPONENTS])
        indices.append(item["row_index"])
        targets.append(item["target"].astype(float))
        predictions.append(matrix @ WEIGHTS)
    index = np.concatenate(indices)
    residual = np.concatenate(targets) - np.concatenate(predictions)
    global_shift = float(residual.mean())
    centered = residual - global_shift
    frame = raw.loc[index]
    artifact = {
        "global_shift": global_shift,
        "count_columns": ["balls_before", "strikes_before"],
        "count_weight": 0.75,
        "count_lookup": make_lookup(
            frame, centered, ["balls_before", "strikes_before"], 500
        ),
        "pitcher_count_columns": ["pitcher_id", "balls_before", "strikes_before"],
        "pitcher_count_weight": 0.25,
        "pitcher_count_lookup": make_lookup(
            frame, centered, ["pitcher_id", "balls_before", "strikes_before"], 300
        ),
    }
    output = Path("artifacts/v14_segment_calibration.joblib")
    joblib.dump(artifact, output, compress=3)
    print(f"Saved {output}, shift={global_shift:.12f}")


if __name__ == "__main__":
    main()
