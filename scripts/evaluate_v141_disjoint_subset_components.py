"""V141: build a component for independence rather than accuracy, using disjoint features.

V138 established the mechanism and the leaderboard confirmed it. The factorization
network won part of the Context slot with a 2024 standalone skill of +147 against
Context's +725, because its correlation with the rest was 0.710 against Context's 0.817.
V137's ladder made the point sharper still: latent 6 scored better standalone on all
three seasons and *failed*, while the weaker latent 8 passed, on independence alone.
V140 then found the limit -- latent 16 and 24 are more independent yet, 0.659 and 0.598,
and both fail because 2022 collapses. So the slot rewards independence up to the point
where the component stops being able to predict at all.

Every component in the blend is built the same way: identity and situation together,
over one shared feature frame. Their correlations sit at 0.854-0.875 for that reason. A
model restricted to a *disjoint* slice of the columns is orthogonal by construction
rather than by luck, which is a way to aim at independence directly instead of hoping a
architecture change produces it.

Two slices, from the official columns only:

`situation` gets the count, the bases, the outs, the inning, the score, the win
expectancies and the leverage index -- and no identity and no as-of history at all. It
cannot know who is pitching, so its errors cannot be the identity errors every other
component makes. The calibration layer already holds count and pitcher-count lookups, so
part of this structure is captured; a model can carry the interactions a lookup cannot.

`identity` is the complement: the pitcher, the batter, their teams and hands, and the
whole as-of block, with no situation. It is the mirror test, and the more likely of the
two to be redundant, since identity is what the existing components are mostly made of.

Both are funded from Context, which V135/V139 show is still the over-weighted component
and which sits at 0.14 after V138 took 0.07 of it.

Ranked by the weakest season, per the four-result record: the three-season average's
reliability falls monotonically with heterogeneity (0.82, 1.82, 0.14 at heterogeneity
7.2, 4.1, 107) while the minimum has under-predicted within a 2.6x-5.5x band every time.

Pre-registered gate, unchanged since V132:
  * three-season equally weighted average CI low > 0
  * three-season average >= +3 points
  * each of 2022, 2023, 2024 mean >= 0
  * monthly block win rate >= 75%
"""

import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v137_context_slot_replacement import NAMES6, blend6, three_season
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from inseason_asof_features_v92 import add_training_inseason_features, drift_correction


OUTPUT = Path("artifacts/v141_disjoint_subset_metrics.json")
PREDICTIONS = Path("artifacts/v141_disjoint_subset_predictions.joblib")
# V138 weights; the new component is funded from Context.
BASE = dict(zip(NAMES6, (0.00, 0.32, 0.14, 0.20, 0.27, 0.07)))
HANDOVERS = (0.03, 0.05, 0.07, 0.10, 0.14)
CONFIG = dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0)
CATBOOST_SOURCE = "no_te_strong"
FACTORIZATION_SOURCE = "latent8"
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
BLOCK_FLOOR = 0.75
AVERAGE_FLOOR = 3.0
P = 100000.0 / 0.25

# Identifiers are categorical no matter what dtype they arrive in. They are numeric in
# the raw file, so a dtype test alone left the identity subset with zero categorical
# columns and CatBoost reading `pitcher_id` as a magnitude -- the one interpretation
# `docs/08` rules out, and it would have made the identity arm a test of the wrong thing.
IDENTIFIERS = ("pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
               "pitcher_hand", "batter_hand")
SITUATION = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li",
]


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    raw_frame = data.drop(columns="row_id")

    situation = [c for c in SITUATION if c in raw_frame.columns]
    identity = [c for c in raw_frame.columns if c not in situation]
    missing = [c for c in SITUATION if c not in raw_frame.columns]
    assert not missing, missing
    assert not set(situation) & set(identity)
    assert set(situation) | set(identity) == set(raw_frame.columns)
    print(f"situation: {len(situation)} columns, identity: {len(identity)} columns "
          f"(disjoint, covering all {len(raw_frame.columns)})", flush=True)
    print(f"  identity holds: {identity[:6]} ...", flush=True)

    subsets = {"situation": situation, "identity": identity}
    frames, categoricals = {}, {}
    for name, columns in subsets.items():
        work = raw_frame[columns].copy()
        cats = [c for c in columns
                if c in IDENTIFIERS or not pd.api.types.is_numeric_dtype(work[c])]
        for column in cats:
            work[column] = work[column].astype(str)
        numeric = [c for c in columns if c not in cats]
        work[numeric] = work[numeric].astype(np.float32)
        frames[name] = work
        categoricals[name] = cats
        print(f"  {name}: {len(cats)} categorical ({cats})", flush=True)

    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()
    predictions, standalone = {}, {}
    for name in subsets:
        predictions[name], standalone[name] = {}, {}
        for year in YEARS:
            started = time.time()
            model = CatBoostClassifier(**CONFIG, random_seed=42, verbose=0,
                                       thread_count=6,
                                       cat_features=categoricals[name],
                                       allow_writing_files=False)
            model.fit(frames[name].loc[season < year], targets[season < year])
            prediction = model.predict_proba(frames[name].loc[season == year])[:, 1]
            predictions[name][str(year)] = prediction
            actual = targets[season == year].astype(float)
            rate = actual.mean()
            standalone[name][str(year)] = float(
                100000 * (1 - ((prediction - actual) ** 2).mean() / (rate * (1 - rate))))
            print(f"  {name} {year}: standalone {standalone[name][str(year)]:8.0f}  "
                  f"[{time.time() - started:.0f}s]", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][20.0]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    catboost = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][CATBOOST_SOURCE]
    factorization = joblib.load(
        "artifacts/v130c_interaction_network_predictions.joblib")["predictions"]
    common = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network, "catboost": catboost,
        "factorization": factorization[FACTORIZATION_SOURCE],
    }
    peers = {k: common[k] for k in ("form", "network", "catboost", "factorization")}
    print("\nmean correlation with the V138 peers (Context is 0.919/0.903/0.817):",
          flush=True)
    for name in subsets:
        cells = []
        for year in YEARS:
            c = np.mean([np.corrcoef(predictions[name][str(year)], p[str(year)])[0, 1]
                         for p in peers.values()])
            cells.append(f"{year} {c:.3f}")
        print(f"  {name:10s} " + "  ".join(cells), flush=True)

    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)
    validation_frame = make_validation_frame()

    # The new component occupies a seventh position; the blend helper takes six, so the
    # extra one is folded in by replacing Context's series with a weighted mixture and
    # keeping the arithmetic explicit.
    def build(slot, handover):
        context_weight = BASE["context"] - handover
        assert context_weight >= -1e-12, handover
        merged = {}
        for year in YEARS:
            key = str(year)
            total = context_weight + handover
            merged[key] = ((context_weight * common["context"][key]
                            + handover * slot[key]) / total if total > 0
                           else common["context"][key])
        return merged, BASE["context"]

    baseline = np.clip(
        blend6(tuple(BASE[n] for n in NAMES6), common, oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(merged, context_weight):
        parts = dict(common, context=merged)
        weights = tuple(context_weight if n == "context" else BASE[n] for n in NAMES6)
        candidate = np.clip(
            blend6(weights, parts, oof, raw_frame) + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate,
            (validation_frame["season"] == 2024).to_numpy())
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        metrics["three_season"] = three_season(metrics)
        metrics["min_season_points"] = min(metrics["three_season"]["season_points"])
        return metrics

    def passes(m):
        t = m["three_season"]
        return (t["ci95_low_points"] > 0 and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= -1e-9 for v in t["season_points"])
                and m["monthly_block_win_rate"] >= BLOCK_FLOOR)

    results = {}
    print(f"\n{'candidate':>22} {'ctx':>5} {'new':>5} {'min':>7} {'avg':>7} "
          f"{'2022':>7} {'2023':>8} {'2024':>7} {'blocks':>7} {'pass':>5}")
    for name in subsets:
        for handover in HANDOVERS:
            merged, context_weight = build(predictions[name], handover)
            label = f"{name}_h{handover:.2f}"
            results[label] = evaluate(merged, context_weight)
            results[label]["handover"] = handover
            results[label]["subset"] = name
            m = results[label]; t = m["three_season"]
            print(f"{label:>22} {BASE['context'] - handover:5.2f} {handover:5.2f} "
                  f"{m['min_season_points']:7.2f} {t['average_points']:7.2f} "
                  f"{t['season_points'][0]:7.2f} {t['season_points'][1]:8.2f} "
                  f"{t['season_points'][2]:7.2f} {m['monthly_block_win_rate']:7.0%} "
                  f"{'YES' if passes(m) else '-':>5}", flush=True)

    eligible = [l for l in results if passes(results[l])]
    promoted = max(eligible, key=lambda l: results[l]["min_season_points"], default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V141_disjoint_subset_components",
        "baseline": "V138 (0.00 / 0.32 / 0.14 / 0.20 / 0.27 / 0.07), Public 1050.5511",
        "rationale": (
            "V138 and V140 together show the slot rewards independence up to the point "
            "the component stops predicting: latent 8 (corr 0.710, 2024 skill +147) beat "
            "both Context (0.817, +725) and the stronger latent 6, while latents 16 and "
            "24 (0.659, 0.598) fail because 2022 collapses. Every existing component "
            "mixes identity and situation over one frame, which is why they all sit at "
            "0.854-0.875. A disjoint column slice is orthogonal by construction."
        ),
        "subsets": {k: v for k, v in subsets.items()},
        "config": CONFIG,
        "handovers": list(HANDOVERS),
        "ranking": "maximum of the weakest season among survivors",
        "standalone_skill": standalone,
        "results": {k: {"subset": v["subset"], "handover": v["handover"],
                        "three_season": v["three_season"],
                        "min_season_points": v["min_season_points"],
                        "monthly_block_win_rate": v["monthly_block_win_rate"]}
                    for k, v in results.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"predictions": predictions}, PREDICTIONS, compress=3)
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
