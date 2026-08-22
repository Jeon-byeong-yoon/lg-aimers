"""V87: rates/hierarchy/middle HGB 3시드를 전체 2019–2024 학습 데이터로 학습."""

from pathlib import Path

import joblib
import pandas as pd

from asof_features_v79 import add_asof_reliability_features, learn_asof_priors
from evaluate_contextual_trackman_v24 import TRACKMAN_COLUMNS
from evaluate_v2 import hist_gbdt_pipeline
from evaluate_v78_bayesian_form_delta import FORM_CONFIG, form_columns
from evaluate_v80_failure_risk_experts import EXPERT_CONFIG, add_expert_deltas
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_prior_season_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_prior_season_target_encodings
from trackman_features import add_trackman_features, prepare_trackman


SEEDS = (42, 1042, 2042)


def train_models(features, y, config):
    models, columns = [], form_columns(features)
    for seed in SEEDS:
        model, model_columns = hist_gbdt_pipeline(features[columns])
        params = {**config, "random_state": seed}
        model.set_params(**{
            f"histgradientboostingclassifier__{name}": value
            for name, value in params.items()
        })
        model.fit(features[model_columns], y)
        # sklearn 1.8 stores a NumPy Generator used only while fitting.
        # NumPy 2.x pickles its PCG64 module path incompatibly with the
        # evaluation server's NumPy 1.26. Prediction does not use this state.
        estimator = model.named_steps["histgradientboostingclassifier"]
        estimator._feature_subsample_rng = None
        models.append(model)
        print(f"  seed={seed} trained", flush=True)
    return models, columns


def main():
    raw = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = raw.pop("control_success").astype("uint8")
    data = raw.drop(columns="row_id")
    target_prior = float(y.mean())
    priors = learn_asof_priors(data)
    encoded = add_prior_season_target_encodings(data, y)
    hierarchical = add_prior_season_hierarchical_encodings(encoded, y, ["pitcher_batter"])
    trackman = prepare_trackman(pd.read_csv(
        "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
    ))

    bundle = {
        "seeds": list(SEEDS), "target_prior": target_prior,
        "asof_priors": priors, "signal_weights": {
            "rates": 0.2, "hierarchy": 0.2, "middle": 0.6,
        },
    }
    for variant in ("rates", "hierarchy"):
        print(f"Training {variant}...", flush=True)
        asof = add_asof_reliability_features(hierarchical, priors, variant)
        form_raw = add_stable_form_features(asof)
        features = select_v2_features(add_row_features(form_raw, target_prior))
        features = add_trackman_features(features, trackman)
        models, columns = train_models(features, y, FORM_CONFIG)
        bundle[f"{variant}_models"] = models
        bundle[f"{variant}_columns"] = columns
        del asof, form_raw, features, models

    print("Training middle...", flush=True)
    middle_features = add_expert_deltas(data, "middle")
    models, columns = train_models(middle_features, y, EXPERT_CONFIG)
    bundle["middle_models"] = models
    bundle["middle_columns"] = columns

    output = Path("artifacts/v87_final_models.joblib")
    joblib.dump(bundle, output, compress=3)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
