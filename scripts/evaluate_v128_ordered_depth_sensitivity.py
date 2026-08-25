"""V128: is ordered boosting's 2022 weakness a small-sample effect?

Ordered boosting produced the largest single-season movement this project has seen:
+232 on 2023 in the blend, and standalone 2023 from -844 to -171. It was blocked on one
condition, 2022 at -59.34, and V125 established that penalty is a genuine per-row
regression rather than an uncalibrated level shift -- the level accounted for only
+1.27 of it.

But "genuine on that fold" is not "genuine for the deployed model". The 2022 fold
trains on 2019-2021, three seasons; the deployed model trains on six. Ordered boosting
is specifically the variant where that should matter. Plain boosting fits each tree on
gradients from the whole training set, whereas ordered boosting fits on prefixes of
random permutations, so early prefixes are short and their estimates noisy. The
mechanism predicts a real dependence on sample size where plain boosting has none.

V121 already measured plain boosting's dependence and found it flat: 757, 748, 765, 745
for one through four seasons. That is the control. If ordered boosting instead climbs
with depth, its 2022 reading is an artifact of the shallowest fold and understates what
a six-season model does, which would put its 2023 signal back in play. If ordered is
also flat, the 2022 regression is a property of the method and the gate was right to
stop it.

Same design as V121 so the two curves are directly comparable: target fixed at 2024,
training window adjacent to it, depth the only thing that varies.
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
from evaluate_v38_lr_grid import v31_form_columns
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from inseason_asof_features_v92 import add_training_inseason_features, feature_names
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


OUTPUT = Path("artifacts/v128_ordered_depth_metrics.json")
PREDICTIONS = Path("artifacts/v128_ordered_depth_predictions.joblib")
FEATURE_SHRINKAGE = 20.0
TARGET = 2024
DEPTHS = (1, 2, 3, 4, 5)
CONFIG = dict(iterations=1200, depth=6, learning_rate=0.02, l2_leaf_reg=12.0,
              boosting_type="Ordered")
# V121's plain-boosting curve, measured under exactly this design.
PLAIN_CURVE = {1: 757, 2: 748, 3: 765, 4: 745, 5: 752}


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

    season = raw_frame["season"].to_numpy()
    targets = y.to_numpy()
    valid_mask = season == TARGET
    valid_target = targets[valid_mask].astype(float)
    rate = valid_target.mean()
    scale = rate * (1 - rate)

    predictions, standalone, rows = {}, {}, {}
    for depth in DEPTHS:
        started = time.time()
        train_mask = (season >= TARGET - depth) & (season < TARGET)
        model = CatBoostClassifier(**CONFIG, random_seed=42, verbose=0, thread_count=6,
                                   cat_features=categorical, allow_writing_files=False)
        model.fit(work.loc[train_mask], targets[train_mask])
        prediction = model.predict_proba(work.loc[valid_mask])[:, 1]
        predictions[str(depth)] = prediction
        standalone[str(depth)] = float(
            100000 * (1 - ((prediction - valid_target) ** 2).mean() / scale))
        rows[str(depth)] = int(train_mask.sum())
        print(f"  ordered depth {depth} ({TARGET - depth}-{TARGET - 1}, "
              f"{train_mask.sum():,} rows): standalone {standalone[str(depth)]:8.0f}  "
              f"(plain was {PLAIN_CURVE[depth]})  [{time.time() - started:.0f}s]",
              flush=True)

    depths = np.array(DEPTHS, dtype=float)
    ordered_skills = np.array([standalone[str(d)] for d in DEPTHS])
    plain_skills = np.array([PLAIN_CURVE[d] for d in DEPTHS], dtype=float)
    ordered_slope = float(np.polyfit(np.log(depths), ordered_skills, 1)[0])
    plain_slope = float(np.polyfit(np.log(depths), plain_skills, 1)[0])
    # Depth 6 is what the deployed model sees; the folds never exceed five.
    ordered_at_6 = float(np.polyval(np.polyfit(np.log(depths), ordered_skills, 1),
                                    np.log(6.0)))
    verdict = ("small-sample effect" if ordered_slope > 3 * abs(plain_slope)
               else "method property")
    print(f"\nslope per log-depth: ordered {ordered_slope:+.0f}, "
          f"plain {plain_slope:+.0f}")
    print(f"projected ordered at depth 6: {ordered_at_6:.0f}")
    print(f"verdict: {verdict}")

    OUTPUT.write_text(json.dumps({
        "experiment": "V128_ordered_depth_sensitivity",
        "question": (
            "Ordered boosting moved 2023 by +232, the largest single-season movement "
            "measured here, and was blocked only by 2022 at -59.34. The 2022 fold trains "
            "on three seasons and the deployed model on six. Ordered boosting fits trees "
            "on prefixes of random permutations, so unlike plain boosting it has a "
            "mechanism that makes sample size matter."
        ),
        "control": (
            "V121 measured plain boosting under this identical design and found it flat "
            "at 757/748/765/745, so any ordered slope is attributable to the method."
        ),
        "target_season": TARGET,
        "depths": list(DEPTHS),
        "training_rows": rows,
        "config": CONFIG,
        "ordered_standalone": standalone,
        "plain_standalone": {str(k): v for k, v in PLAIN_CURVE.items()},
        "ordered_slope_per_log_depth": ordered_slope,
        "plain_slope_per_log_depth": plain_slope,
        "ordered_projected_at_depth_6": ordered_at_6,
        "verdict": verdict,
        "compliance": {"official_data_only": True, "test_csv_read": False,
                       "chronological_folds": True, "fixed_seed": True},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"predictions": predictions}, PREDICTIONS, compress=3)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
