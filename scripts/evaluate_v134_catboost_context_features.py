"""V134: give CatBoost the contextual Trackman columns it has never seen.

Laying the four components side by side exposes a gap:

    component   weight   in-season   contextual Trackman   learner
    Form         0.32       yes             no             HGB, 500 iters, lr 0.03
    Context      0.21       no             yes (35 cols)   HGB, 200 iters, lr 0.06
    network      0.20       yes             no             MLP
    CatBoost     0.27       yes             no             1200 iters, lr 0.02

The 27 `tm_ctx_*` columns and 8 `tm_*_std` columns are seen by exactly one component,
and it is the oldest and most lightly fitted learner in the ensemble -- 200 boosting
iterations at learning rate 0.06, chosen in V31 before the blend was restructured three
times. Meanwhile the learner that has beaten everything else here has never been handed
those columns at all. Giving the strongest learner a feature set it has not seen is the
one lever left that has not been pulled.

The Context model's 0.21 weight is itself the evidence that these columns carry signal:
V119's ladder put that axis's vertex within 0.015 of its current value, so the component
is earning its place. The question is whether a 200-iteration HGB is extracting what is
there.

Feature construction follows V116/V117 exactly and adds `add_context_trackman_features`,
keeping the `no_te` treatment that won in V116 -- `te_*` and `hte_*` removed so CatBoost
builds its own ordered target statistics rather than consuming ours. The global target
prior is used rather than a per-fold one, matching what the deployed script reads from
the frozen bundle.

Second stage: if these columns help inside CatBoost, the Context component may be
partly redundant, so its weight is laddered for the winning variant. That is the whole
point of asking -- not to add capability twice, but to put it where it works best.

Pre-registered gate, unchanged from V132:
  * three-season equally weighted average CI low > 0
  * three-season average >= +3 points
  * each of 2022, 2023, 2024 mean >= 0
  * monthly block win rate >= 75%
"""

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")
from contextual_trackman_v24 import add_context_trackman_features, prepare_context_trackman
from embedding_network_v111 import EMBEDDING_SPECS
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v112_network_weight_and_season import bootstrap
from evaluate_v119_five_way_weight_refit import NAMES, blend
from evaluate_v38_lr_grid import v31_form_columns
from evaluate_v77_v41_error_diagnostics import YEARS
from evaluate_v88_transfer_validation import make_validation_frame
from evaluate_v89_recency_sample_weight import development_metrics
from evaluate_residual_ridge_v13 import raw_prediction as v11_prediction
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import (
    add_training_inseason_features, drift_correction, feature_names,
)
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v134_catboost_context_metrics.json")
PREDICTIONS = Path("artifacts/v134_catboost_context_predictions.joblib")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
BASE = (0.00, 0.32, 0.21, 0.20, 0.27)
INCUMBENT = "no_te_strong"
# One config only. The first attempt at two was killed by the operating system part
# way through its first fit: the eleven stringified categoricals are object dtype over
# 1.475M rows, every `.loc` slice copies them, and 27 extra numeric columns pushed the
# peak past what this machine has. `strong` is the config V116 selected, so it is the
# one worth the memory.
CONFIGS = {
    "strong": dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0),
}
CONTEXT_LADDER = (0.09, 0.13, 0.17, 0.21, 0.25)
BLOCK_FLOOR = 0.75
AVERAGE_FLOOR = 3.0
P = 100000.0 / 0.25


def three_season(metrics):
    seasons = metrics["season_bootstrap"]
    means = [seasons[str(y)]["mean"] for y in YEARS]
    errors = [(seasons[str(y)]["ci95_high"] - seasons[str(y)]["ci95_low"]) / 3.9199
              for y in YEARS]
    average = sum(means) / 3.0
    combined = math.sqrt(sum(e * e for e in errors)) / 3.0
    return {"average_points": average * P, "se_points": combined * P,
            "ci95_low_points": (average - 1.96 * combined) * P,
            "season_points": [m * P for m in means]}


def main():
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    raw_frame = data.copy()

    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    form_raw = add_stable_form_features(hierarchical)
    block = add_training_inseason_features(
        raw_frame, shrinkage=FEATURE_SHRINKAGE)[feature_names()]
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS))
    # Takes the already-prepared frame, not the raw CSV: the pitch-group dummies it
    # aggregates are created by `prepare_trackman`.
    context_trackman = prepare_context_trackman(trackman)

    features = select_v2_features(add_row_features(form_raw, float(y.mean())))
    features = add_trackman_features(features, trackman)
    baseline_columns = set(v31_form_columns(pd.concat(
        [features, block], axis=1)))
    features = add_context_trackman_features(features, context_trackman)
    features = pd.concat([features, block], axis=1)
    model_columns = v31_form_columns(features)
    added = [c for c in model_columns if c not in baseline_columns]

    categorical = [name for name, _ in EMBEDDING_SPECS if name in model_columns]
    for name, _ in EMBEDDING_SPECS:
        if name not in features:
            features[name] = raw_frame[name]
            if name not in categorical:
                categorical.append(name)
    encoding_columns = [c for c in model_columns if c.startswith(("te_", "hte_"))]
    columns = ([c for c in model_columns if c not in encoding_columns]
               + [c for c in categorical if c not in model_columns])
    work = features[columns].copy()
    for name in categorical:
        work[name] = work[name].astype(str)
    numeric = [c for c in columns if c not in categorical]
    work[numeric] = work[numeric].astype(np.float32)
    print(f"frame {work.shape[0]:,} x {work.shape[1]} columns "
          f"(V117 used 96); {len(added)} contextual columns added", flush=True)
    print(f"  added: {sorted(added)[:6]} ...", flush=True)

    # Everything above is dead once `work` exists, and holding it alongside CatBoost's
    # own copies is what killed the first attempt.
    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()
    del (features, hierarchical, encoded, form_raw, block, trackman, context_trackman,
         data, baseline_columns)
    gc.collect()
    print(f"  resident after cleanup: "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GiB",
          flush=True)

    predictions, standalone = {}, {}
    for name, config in CONFIGS.items():
        key = f"ctx_{name}"
        predictions[key], standalone[key] = {}, {}
        for year in YEARS:
            started = time.time()
            model = CatBoostClassifier(**config, random_seed=42, verbose=0,
                                       thread_count=6, cat_features=categorical,
                                       allow_writing_files=False)
            train_slice = work.loc[season < year]
            model.fit(train_slice, targets[season < year])
            del train_slice
            gc.collect()
            prediction = model.predict_proba(work.loc[season == year])[:, 1]
            del model
            gc.collect()
            predictions[key][str(year)] = prediction
            actual = targets[season == year].astype(float)
            rate = actual.mean()
            standalone[key][str(year)] = float(
                100000 * (1 - ((prediction - actual) ** 2).mean() / (rate * (1 - rate))))
            print(f"  {key} {year}: standalone {standalone[key][str(year)]:8.0f}  "
                  f"[{time.time() - started:.0f}s, peak "
                  f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} "
                  f"GiB]", flush=True)

    oof = joblib.load("artifacts/v6_oof_predictions.joblib")
    logistic = joblib.load("artifacts/v17_logistic_predictions.joblib")["0.3"]
    form = joblib.load("artifacts/v102_inseason_smoothing_predictions.joblib")
    form = form["forms"][FEATURE_SHRINKAGE]
    context = joblib.load("artifacts/v31_feature_removal_predictions.joblib")
    context = context["no_matchup_hte"]["context"]
    network = joblib.load("artifacts/v112_network_weight_predictions.joblib")
    network = network["networks"]["without_season"]
    incumbent = joblib.load(
        "artifacts/v116_catboost_predictions.joblib")["predictions"][INCUMBENT]
    common = {
        "v17": {str(v): 0.95 * v11_prediction(oof[str(v)]) + 0.05 * logistic[str(v)]
                for v in YEARS},
        "form": form, "context": context, "network": network,
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    baseline = np.clip(
        blend(BASE, dict(common, catboost=incumbent), oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    def evaluate(slot, weights):
        candidate = np.clip(
            blend(weights, dict(common, catboost=slot), oof, raw_frame)
            + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate,
            (validation_frame["season"] == 2024).to_numpy())
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        metrics["three_season"] = three_season(metrics)
        metrics["weights"] = {n: round(w, 4) for n, w in zip(NAMES, weights)}
        return metrics

    def passes(metrics):
        t = metrics["three_season"]
        return (t["ci95_low_points"] > 0 and t["average_points"] >= AVERAGE_FLOOR
                and all(v >= -1e-9 for v in t["season_points"])
                and metrics["monthly_block_win_rate"] >= BLOCK_FLOOR)

    def show(label, metrics):
        t = metrics["three_season"]
        print(f"  {label:22s} avg {t['average_points']:+7.2f}  "
              f"CIlo {t['ci95_low_points']:+7.2f}  2022 {t['season_points'][0]:+7.2f}  "
              f"2023 {t['season_points'][1]:+8.2f}  2024 {t['season_points'][2]:+7.2f}  "
              f"blocks {metrics['monthly_block_win_rate']:5.0%}  "
              f"{'YES' if passes(metrics) else '-'}", flush=True)

    print("\nslot swap at weight 0.27 (baseline = V122's no_te_strong):", flush=True)
    stage_one = {}
    for key, slot in predictions.items():
        stage_one[key] = evaluate(slot, BASE)
        show(key, stage_one[key])

    best = max(stage_one, key=lambda k: stage_one[k]["three_season"]["ci95_low_points"])
    print(f"\nbest: {best}. Laddering the Context weight -- if CatBoost now reads these "
          f"columns, Context may be partly redundant.", flush=True)
    ladder = {}
    for weight in CONTEXT_LADDER:
        form_weight = round(BASE[1] + (BASE[2] - weight), 4)
        if form_weight <= 0:
            continue
        weights = (BASE[0], form_weight, round(weight, 4), BASE[3], BASE[4])
        label = f"{best}_ctx{weight:.2f}"
        ladder[label] = evaluate(predictions[best], weights)
        ladder[label]["context_weight"] = weight
        show(label, ladder[label])

    combined = {**{f"slot:{k}": v for k, v in stage_one.items()},
                **{f"ladder:{k}": v for k, v in ladder.items()}}
    eligible = [k for k, v in combined.items() if passes(v)]
    promoted = max(eligible,
                   key=lambda k: combined[k]["three_season"]["ci95_low_points"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V134_catboost_context_features",
        "baseline": "V122 (0.00 / 0.32 / 0.21 / 0.20 / 0.27), Public 1047.03653",
        "rationale": (
            "The 27 tm_ctx_* and 8 tm_*_std columns are read by exactly one component, "
            "the oldest and most lightly fitted learner in the ensemble (200 boosting "
            "iterations at lr 0.06, chosen in V31 before three blend restructurings). "
            "CatBoost, which has beaten everything else here, has never been handed "
            "them."
        ),
        "columns_added": sorted(added),
        "frame_columns": int(work.shape[1]),
        "configs": CONFIGS,
        "context_ladder": list(CONTEXT_LADDER),
        "gate": {"three_season_ci95_low_points": "> 0",
                 "three_season_average_points": f">= {AVERAGE_FLOOR}",
                 "each_season_mean_points": ">= 0",
                 "monthly_block_win_rate": f">= {BLOCK_FLOOR}"},
        "incumbent_standalone": {"2022": 2397, "2023": -844, "2024": 752},
        "standalone_skill": standalone,
        "stage_one": {k: {"three_season": v["three_season"],
                          "monthly_block_win_rate": v["monthly_block_win_rate"]}
                      for k, v in stage_one.items()},
        "best_variant": best,
        "ladder": {k: {"context_weight": v["context_weight"], "weights": v["weights"],
                       "three_season": v["three_season"],
                       "monthly_block_win_rate": v["monthly_block_win_rate"]}
                   for k, v in ladder.items()},
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"predictions": predictions}, PREDICTIONS, compress=3)

    print("\nstandalone (incumbent: 2022 2397, 2023 -844, 2024 752):")
    for key, values in standalone.items():
        print(f"  {key:12s} " + "  ".join(f"{k} {v:8.0f}" for k, v in values.items()))
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
