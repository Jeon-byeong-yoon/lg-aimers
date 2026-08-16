"""Evaluate leakage-safe segment residual calibration on top of V11."""

import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


COMPONENTS = ["extra_trees", "trackman_hgb", "te_trackman_hgb", "hierarchical_hgb"]
WEIGHTS = np.array([
    0.29483562599237795,
    0.2344456574665245,
    0.11617615437521091,
    0.35454256216588664,
])
GROUPS = {
    "count": ["balls_before", "strikes_before"],
    "hand_matchup": ["pitcher_hand", "batter_hand"],
    "count_hand": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
    "base_state": ["base_state"],
    "outs": ["outs_before"],
    "inning": ["inning_bucket"],
    "month": ["game_month"],
    "game_type": ["game_type"],
    "pitcher": ["pitcher_id"],
    "batter": ["batter_id"],
    "pitcher_count": ["pitcher_id", "balls_before", "strikes_before"],
}
SMOOTHING = [50, 100, 300, 1000, 3000]
ALPHAS = [0.25, 0.5, 0.75, 1.0]


def add_features(frame):
    output = frame.copy()
    output["inning_bucket"] = pd.cut(
        output["inning"], bins=[0, 3, 6, 9, np.inf], labels=False, include_lowest=True
    ).astype("int8")
    return output


def prediction(item):
    matrix = np.column_stack([item["components"][name] for name in COMPONENTS])
    return matrix @ WEIGHTS


def segment_correction(train_frame, train_residual, valid_frame, columns, smoothing):
    work = train_frame[columns].copy()
    work["_residual"] = train_residual - train_residual.mean()
    stats = work.groupby(columns, dropna=False)["_residual"].agg(["sum", "count"])
    stats["_correction"] = stats["sum"] / (stats["count"] + smoothing)
    if len(columns) == 1:
        mapping = stats["_correction"]
        return valid_frame[columns[0]].map(mapping).fillna(0).to_numpy()
    lookup = stats[["_correction"]].reset_index()
    merged = valid_frame[columns].merge(lookup, how="left", on=columns, sort=False)
    return merged["_correction"].fillna(0).to_numpy()


def evaluate_fold(oof, frame, history_years, valid_year):
    history_items = [oof[str(year)] for year in history_years]
    train_y = np.concatenate([item["target"] for item in history_items]).astype(float)
    train_p = np.concatenate([prediction(item) for item in history_items])
    train_index = np.concatenate([item["row_index"] for item in history_items])
    valid_item = oof[str(valid_year)]
    valid_y = valid_item["target"].astype(float)
    valid_p = prediction(valid_item)
    valid_index = valid_item["row_index"]
    global_shift = float((train_y - train_p).mean())
    baseline_p = np.clip(valid_p + global_shift, 0, 1)
    baseline = float(brier_score_loss(valid_y, baseline_p))
    corrections = {}
    results = {}
    for name, columns in GROUPS.items():
        for smoothing in SMOOTHING:
            correction = segment_correction(
                frame.loc[train_index], train_y - train_p,
                frame.loc[valid_index], columns, smoothing,
            )
            corrections[(name, smoothing)] = correction
            for alpha in ALPHAS:
                score = float(brier_score_loss(
                    valid_y, np.clip(baseline_p + alpha * correction, 0, 1)
                ))
                results[f"{name}|s={smoothing}|a={alpha}"] = score
    return baseline, results, corrections, valid_y, baseline_p


def main():
    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    frame = add_features(raw.drop(columns=["row_id", "control_success"]))
    base23, res23, corr23, y23, pred23 = evaluate_fold(oof, frame, [2022], 2023)
    base24, res24, corr24, y24, pred24 = evaluate_fold(oof, frame, [2022, 2023], 2024)

    accepted = []
    for key in sorted(set(res23) & set(res24)):
        if res23[key] < base23 and res24[key] < base24:
            accepted.append({
                "method": key,
                "2023_brier": res23[key],
                "2024_brier": res24[key],
                "2023_gain": base23 - res23[key],
                "2024_gain": base24 - res24[key],
            })

    # Also test equal mixtures of the strongest distinct single-group corrections.
    ranked_names = []
    for item in sorted(accepted, key=lambda x: min(x["2023_gain"], x["2024_gain"]), reverse=True):
        name = item["method"].split("|")[0]
        if name not in ranked_names:
            ranked_names.append(name)
        if len(ranked_names) == 6:
            break
    by_name = {}
    for item in accepted:
        name = item["method"].split("|")[0]
        if name in ranked_names and name not in by_name:
            candidates = [x for x in accepted if x["method"].split("|")[0] == name]
            by_name[name] = max(candidates, key=lambda x: min(x["2023_gain"], x["2024_gain"]))
    mixtures = []
    for left, right in combinations(by_name, 2):
        left_key, right_key = by_name[left]["method"], by_name[right]["method"]
        def parse(key):
            parts = key.split("|")
            return parts[0], int(parts[1][2:]), float(parts[2][2:])
        ln, ls, la = parse(left_key)
        rn, rs, ra = parse(right_key)
        p23 = pred23 + 0.5 * (la * corr23[(ln, ls)] + ra * corr23[(rn, rs)])
        p24 = pred24 + 0.5 * (la * corr24[(ln, ls)] + ra * corr24[(rn, rs)])
        score23 = float(brier_score_loss(y23, np.clip(p23, 0, 1)))
        score24 = float(brier_score_loss(y24, np.clip(p24, 0, 1)))
        if score23 < base23 and score24 < base24:
            mixtures.append({
                "method": f"mix({left_key},{right_key})",
                "2023_brier": score23, "2024_brier": score24,
                "2023_gain": base23 - score23, "2024_gain": base24 - score24,
            })

    all_candidates = accepted + mixtures
    all_candidates.sort(key=lambda x: (min(x["2023_gain"], x["2024_gain"]),
                                        x["2023_gain"] + x["2024_gain"]), reverse=True)
    result = {
        "baseline": {"2023_brier": base23, "2024_brier": base24},
        "accepted_single_count": len(accepted),
        "accepted_mixture_count": len(mixtures),
        "best": all_candidates[0] if all_candidates else None,
        "top10": all_candidates[:10],
    }
    Path("artifacts/v12_segment_calibration_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
