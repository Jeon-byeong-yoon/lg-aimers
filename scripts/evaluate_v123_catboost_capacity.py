"""V123: CatBoost capacity and categorical-handling variants, since more data does not help.

V121 varied training depth while holding the target at 2024 and the window adjacent to
it, which is exactly the fold-versus-deployed contrast. Standalone skill came out
757 / 748 / 765 / 745 for one through four seasons -- flat inside noise. One season of
245,525 rows is as good as four of 984,172.

That rules out the explanation offered for V117's 2.59x over-delivery, and it says
something more useful: **the component is not sample-limited**. Adding rows does
nothing, so the binding constraint is capacity or how the categoricals are handled,
not data volume. It also fits the league drift already documented here, where the
control-success rate fell monotonically from 0.5647 to 0.4861; older seasons describe
a measurably different game, so their rows dilute the ordered statistics as much as
they inform them.

Four variants follow from that, all against the V122 blend at catboost 0.27:

`deep8` doubles the tree depth. If capacity binds, this is the most direct test.

`long` doubles the iterations at half the learning rate, the same direction that made
`strong` beat `gentle` in V116, pushed further.

`ordered` switches to CatBoost's ordered boosting. This is the library's distinctive
mechanism against target leakage and it has never been tried here; the default
`Plain` mode uses ordered target statistics but plain gradient estimation.

`onehot` raises `one_hot_max_size` to 16, which one-hot encodes the twelve count
states, eight base states, thirteen team codes and the hand columns, leaving ordered
target statistics for `pitcher_id` and `batter_id` alone. If the statistics help only
where cardinality is genuinely high, restricting them there should be neutral or
better, and it isolates which part of CatBoost's categorical machinery is doing the
work that LightGBM's could not do in V118.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, "scripts")
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


OUTPUT = Path("artifacts/v123_catboost_capacity_metrics.json")
PREDICTIONS = Path("artifacts/v123_catboost_capacity_predictions.joblib")
FEATURE_SHRINKAGE = 20.0
DRIFT_SHRINKAGE = 3.0
DRIFT_WEIGHT = 0.10
# V122 weights; the catboost slot is what these variants compete for.
BASE = (0.00, 0.32, 0.21, 0.20, 0.27)
INCUMBENT = "no_te_strong"
CONFIGS = {
    "deep8": dict(iterations=1200, depth=8, learning_rate=0.02, l2_leaf_reg=12.0),
    "long": dict(iterations=2400, depth=6, learning_rate=0.01, l2_leaf_reg=12.0),
    "ordered": dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0,
                    boosting_type="Ordered"),
    "onehot": dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0,
                   one_hot_max_size=16),
}
P = 100000.0 / 0.25


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

    features = select_v2_features(add_row_features(form_raw, float(y.mean())))
    features = add_trackman_features(features, trackman)
    features = pd.concat([features, block], axis=1)
    model_columns = v31_form_columns(features)
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
    print(f"frame {work.shape[0]:,} x {work.shape[1]}  categorical {len(categorical)}",
          flush=True)

    targets = y.to_numpy()
    season = raw_frame["season"].to_numpy()
    predictions, standalone = {}, {}
    for name, config in CONFIGS.items():
        predictions[name], standalone[name] = {}, {}
        for year in YEARS:
            started = time.time()
            train_mask = season < year
            valid_mask = season == year
            model = CatBoostClassifier(**config, random_seed=42, verbose=0,
                                       thread_count=6, cat_features=categorical,
                                       allow_writing_files=False)
            model.fit(work.loc[train_mask], targets[train_mask])
            prediction = model.predict_proba(work.loc[valid_mask])[:, 1]
            predictions[name][str(year)] = prediction
            target = targets[valid_mask].astype(float)
            rate = target.mean()
            standalone[name][str(year)] = float(
                100000 * (1 - ((prediction - target) ** 2).mean() / (rate * (1 - rate))))
            print(f"  {name} {year}: standalone {standalone[name][str(year)]:8.0f}  "
                  f"[{time.time() - started:.0f}s]", flush=True)

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
        "v17": {str(y): 0.95 * v11_prediction(oof[str(y)]) + 0.05 * logistic[str(y)]
                for y in YEARS},
        "form": form, "context": context, "network": network,
    }
    order = np.concatenate([oof[str(year)]["row_index"] for year in YEARS])
    term = drift_correction(
        add_training_inseason_features(
            raw_frame, shrinkage=DRIFT_SHRINKAGE).loc[order], 1.0)

    validation_frame = make_validation_frame()
    mask_2024 = (validation_frame["season"] == 2024).to_numpy()
    baseline = np.clip(
        blend(BASE, dict(common, catboost=incumbent), oof, raw_frame)
        + DRIFT_WEIGHT * term, 0, 1)
    reference = validation_frame.copy()
    reference["v41_prediction"] = baseline
    reference["v41_squared_error"] = (baseline - reference["target"]) ** 2

    results = {}
    for name in CONFIGS:
        candidate = np.clip(
            blend(BASE, dict(common, catboost=predictions[name]), oof, raw_frame)
            + DRIFT_WEIGHT * term, 0, 1)
        metrics = development_metrics(reference, candidate)
        metrics["bootstrap_2024"] = bootstrap(
            validation_frame, baseline, candidate, mask_2024)
        metrics["season_bootstrap"] = {
            str(year): bootstrap(validation_frame, baseline, candidate,
                                 (validation_frame["season"] == year).to_numpy())
            for year in YEARS}
        metrics["config"] = CONFIGS[name]
        results[name] = metrics
        b = metrics["bootstrap_2024"]; s = metrics["season_bootstrap"]
        print(f"  {name:9s} CIlo {b['ci95_low']*P:+7.2f}  mean {b['mean']*P:+7.2f}  "
              f"2022 {s['2022']['mean']*P:+7.2f}  2023 {s['2023']['mean']*P:+7.2f}  "
              f"blocks {metrics['monthly_block_win_rate']:5.0%}", flush=True)

    def passes(label):
        r = results[label]
        return (r["bootstrap_2024"]["ci95_low"] > 0
                and r["season_bootstrap"]["2022"]["mean"] > -1e-5
                and r["season_bootstrap"]["2023"]["mean"] > -1e-5
                and r["monthly_block_win_rate"] >= 0.75
                and r["bootstrap_2024"]["mean"] * P >= 3.0)

    eligible = [label for label in results if passes(label)]
    promoted = max(eligible, key=lambda l: results[l]["bootstrap_2024"]["ci95_low"],
                   default=None)

    OUTPUT.write_text(json.dumps({
        "experiment": "V123_catboost_capacity",
        "baseline": "V122 (0.00 / 0.32 / 0.21 / 0.20 / 0.27) with no_te_strong",
        "rationale": (
            "V121 showed standalone skill flat in training depth (757/748/765/745 for "
            "one to four seasons), so the component is not sample-limited and the "
            "binding constraint is capacity or categorical handling."
        ),
        "incumbent_standalone": {"2022": 2397, "2023": -844, "2024": 752},
        "configs": CONFIGS,
        "standalone_skill": standalone,
        "results": results,
        "eligible_candidates": sorted(eligible),
        "promoted_candidate": promoted,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"predictions": predictions}, PREDICTIONS, compress=3)

    print("\nstandalone skill (incumbent: 2022 2397, 2023 -844, 2024 752):")
    for name, values in standalone.items():
        print(f"  {name:9s} " + "  ".join(f"{k} {v:8.0f}" for k, v in values.items()))
    print(f"\neligible={len(eligible)}  promoted={promoted}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
