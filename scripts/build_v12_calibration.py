"""Build fixed V12 segment calibration lookups from chronological OOF only."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


COMPONENTS = ["extra_trees", "trackman_hgb", "te_trackman_hgb", "hierarchical_hgb"]
WEIGHTS = np.array([
    0.29483562599237795, 0.2344456574665245,
    0.11617615437521091, 0.35454256216588664,
])


def make_lookup(frame, centered_residual, columns, smoothing):
    work = frame[columns].copy()
    work["_residual"] = centered_residual
    stats = work.groupby(columns, dropna=False)["_residual"].agg(["sum", "count"])
    stats["correction"] = stats["sum"] / (stats["count"] + smoothing)
    return stats[["correction"]].reset_index()


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
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    residual = target - prediction
    global_shift = float(residual.mean())
    centered = residual - global_shift
    frame = raw.loc[index]
    artifact = {
        "global_shift": global_shift,
        "count_columns": ["balls_before", "strikes_before"],
        "count_weight": 0.5,
        "count_lookup": make_lookup(
            frame, centered, ["balls_before", "strikes_before"], 50
        ),
        "pitcher_count_columns": ["pitcher_id", "balls_before", "strikes_before"],
        "pitcher_count_weight": 0.125,
        "pitcher_count_lookup": make_lookup(
            frame, centered, ["pitcher_id", "balls_before", "strikes_before"], 300
        ),
    }
    output = Path("artifacts/v12_segment_calibration.joblib")
    joblib.dump(artifact, output, compress=3)
    print(
        f"Saved {output}, shift={global_shift:.12f}, "
        f"count_rows={len(artifact['count_lookup'])}, "
        f"pitcher_count_rows={len(artifact['pitcher_count_lookup'])}"
    )


if __name__ == "__main__":
    main()
