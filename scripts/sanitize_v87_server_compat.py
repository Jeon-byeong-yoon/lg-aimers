"""Remove fit-only NumPy 2.x RNG state from the trained V87 sklearn models."""

from pathlib import Path

import joblib


def main():
    path = Path("artifacts/v87_final_models.joblib")
    bundle = joblib.load(path)
    changed = 0
    for group in ("rates_models", "hierarchy_models", "middle_models"):
        for model in bundle[group]:
            estimator = model.named_steps["histgradientboostingclassifier"]
            if getattr(estimator, "_feature_subsample_rng", None) is not None:
                estimator._feature_subsample_rng = None
                changed += 1
    joblib.dump(bundle, path, compress=3)
    print(f"Sanitized {changed} fit-only RNG objects in {path}")


if __name__ == "__main__":
    main()
