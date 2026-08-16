"""Robustly tune V6 component weights with expanding additive calibration."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss


NAMES = ["extra_trees", "trackman_hgb", "te_trackman_hgb", "hierarchical_hgb"]
BASE_WEIGHTS = np.array([0.40, 0.36, 0.144, 0.096])


def arrays(oof, year):
    item = oof[str(year)]
    matrix = np.column_stack([item["components"][name] for name in NAMES])
    return item["target"].astype(float), matrix


def calibrated_score(train_y, train_x, valid_y, valid_x, weights):
    train_prediction = train_x @ weights
    valid_prediction = valid_x @ weights
    shift = float(train_y.mean() - train_prediction.mean())
    prediction = np.clip(valid_prediction + shift, 0, 1)
    return float(brier_score_loss(valid_y, prediction)), shift


def calibrated_scores(train_y, train_x, valid_y, valid_x, weights):
    """Vectorized un-clipped Brier scores; accepted candidates are rechecked exactly."""
    centered = valid_x - train_x.mean(axis=0)
    residual = valid_y - train_y.mean()
    gram = centered.T @ centered / len(valid_y)
    cross = centered.T @ residual / len(valid_y)
    constant = float(np.mean(residual**2))
    return (
        np.einsum("ij,jk,ik->i", weights, gram, weights)
        - 2 * (weights @ cross)
        + constant
    )


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    y22, x22 = arrays(oof, 2022)
    y23, x23 = arrays(oof, 2023)
    y24, x24 = arrays(oof, 2024)

    base23, base_shift23 = calibrated_score(y22, x22, y23, x23, BASE_WEIGHTS)
    base24, base_shift24 = calibrated_score(
        np.concatenate([y22, y23]), np.vstack([x22, x23]), y24, x24, BASE_WEIGHTS
    )

    # Reproducible local exploration around the accepted blend plus broad simplex draws.
    rng = np.random.default_rng(20260815)
    local = rng.dirichlet(BASE_WEIGHTS * 120, size=12000)
    broad = rng.dirichlet(np.ones(4) * 3, size=3000)
    candidates = np.vstack([BASE_WEIGHTS, local, broad])
    scores23 = calibrated_scores(y22, x22, y23, x23, candidates)
    history_y = np.concatenate([y22, y23])
    history_x = np.vstack([x22, x23])
    scores24 = calibrated_scores(history_y, history_x, y24, x24, candidates)
    viable = np.flatnonzero((scores23 < base23) & (scores24 < base24))
    if len(viable) > 20:
        robust_gain = np.minimum(base23 - scores23[viable], base24 - scores24[viable])
        viable = viable[np.argsort(robust_gain)[-20:]]
    accepted = []
    for index in viable:
        weights = candidates[index]
        # Exact clipped evaluation guards against any boundary discrepancy.
        score23, shift23 = calibrated_score(y22, x22, y23, x23, weights)
        score24, shift24 = calibrated_score(history_y, history_x, y24, x24, weights)
        gain23, gain24 = base23 - score23, base24 - score24
        if gain23 > 0 and gain24 > 0:
            accepted.append((min(gain23, gain24), gain23 + gain24, weights,
                             score23, score24, shift23, shift24))

    accepted.sort(key=lambda row: (row[0], row[1]), reverse=True)
    best = accepted[0] if accepted else None
    result = {
        "baseline": {
            "weights": dict(zip(NAMES, BASE_WEIGHTS.tolist())),
            "2023_brier": base23,
            "2024_brier": base24,
            "2023_shift": base_shift23,
            "2024_shift": base_shift24,
        },
        "candidate_count": int(len(candidates)),
        "accepted_count": len(accepted),
    }
    if best:
        _, _, weights, score23, score24, shift23, shift24 = best
        all_y = np.concatenate([y22, y23, y24])
        all_x = np.vstack([x22, x23, x24])
        final_shift = float(all_y.mean() - (all_x @ weights).mean())
        result["best"] = {
            "weights": dict(zip(NAMES, weights.tolist())),
            "2023_brier": score23,
            "2024_brier": score24,
            "2023_gain": base23 - score23,
            "2024_gain": base24 - score24,
            "2023_shift": shift23,
            "2024_shift": shift24,
            "final_shift": final_shift,
        }
    Path("artifacts/v11_blend_calibration_tuning.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
