"""Build fixed V15 multi-segment calibration from chronological OOF."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_v12_calibration import COMPONENTS, WEIGHTS, make_lookup


SPECS = [
    (["balls_before", "strikes_before"], 500, 0.75, "count"),
    (["pitcher_id", "balls_before", "strikes_before"], 300, 0.25, "pitcher_count"),
    (["base_state"], 10000, 0.5, "base_state"),
    (["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
     50, 0.125, "count_hand"),
]


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
    corrections = []
    for columns, smoothing, weight, name in SPECS:
        corrections.append({
            "name": name,
            "columns": columns,
            "smoothing": smoothing,
            "weight": weight,
            "lookup": make_lookup(frame, centered, columns, smoothing),
        })
    artifact = {"global_shift": global_shift, "corrections": corrections}
    output = Path("artifacts/v15_segment_calibration.joblib")
    joblib.dump(artifact, output, compress=3)
    print(f"Saved {output}, shift={global_shift:.12f}, corrections={len(corrections)}")


if __name__ == "__main__":
    main()
