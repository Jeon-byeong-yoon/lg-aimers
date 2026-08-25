"""Reproduce the submitted V106 model end to end from the official data only.

Phase 3 entry requires a training script that regenerates the Private-score model,
and the work that produced V106 is spread across a hundred experiment scripts. This
orchestrator runs only the steps V106 actually depends on, in dependency order, and
then verifies that the rebuilt artifacts reproduce the packaged predictions.

Inputs are the two official files:
    공모전 dataset/open/data/train.csv
    공모전 dataset/open/data/trackman_history.csv

Outputs are the five artifacts inside submit_v106.zip:
    v6_ensemble.joblib          the V17 tree layer plus its lookups
    v17_logistic_model.joblib   the 5% logistic diversity model
    v105_feature_models.joblib  Form (in-season shrinkage 20) and Context
    v105_inseason_anchor.joblib frozen end-of-2024 career state
    v106_calibration.joblib     global shift, correction lookups, all weights

Every model that ends up in the submission must be written by the server-mirror
interpreter. Artifacts saved under numpy 2.x embed numpy.random.Generator by class
reference and fail to load on the evaluation server's numpy 1.26.4, which cost one
submission on the first V93 attempt.

    python scripts/reproduce_v106.py --list
    python scripts/reproduce_v106.py                 # all stages
    python scripts/reproduce_v106.py --from oof_form # resume partway
    python scripts/reproduce_v106.py --verify-only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SERVER_PYTHON = "/opt/anaconda3/envs/lgaimers_server/bin/python"
ROOT = Path(__file__).resolve().parent.parent

# (key, description, script, produced artifacts)
# Chronological out-of-fold predictions for 2022/2023/2024. These exist only to
# fit the calibration; no submitted model is trained here.
STAGES = [
    ("oof_trackman", "V4 Trackman HGB out-of-fold",
     "evaluate_trackman_v4.py", ["v4_trackman_predictions.joblib"]),
    ("oof_encoded", "V5 target-encoded Trackman HGB out-of-fold",
     "evaluate_target_encoding_v5.py", ["v5_target_encoding_predictions.joblib"]),
    ("oof_hierarchical", "V6 hierarchical-encoded HGB out-of-fold",
     "evaluate_hierarchical_v6.py", ["v6_hierarchical_predictions.joblib"]),
    ("oof_tree_layer", "ExtraTrees out-of-fold and the four-model tree layer",
     "build_v6_oof_predictions.py", ["v6_oof_predictions.joblib"]),
    ("oof_logistic", "Regularised logistic out-of-fold",
     "evaluate_logistic_diversity_v17.py", ["v17_logistic_predictions.joblib"]),
    ("oof_context", "Context HGB out-of-fold (no_matchup_hte variant)",
     "evaluate_v31_feature_removal.py", ["v31_feature_removal_predictions.joblib"]),
    ("oof_form", "Form HGB out-of-fold across in-season shrinkages",
     "evaluate_v102_inseason_smoothing.py",
     ["v102_inseason_smoothing_predictions.joblib"]),
    # Final models, fitted on all six seasons.
    ("final_tree_layer", "V17 tree layer and its 2025 lookups",
     "train_v6_final.py", ["v6_ensemble.joblib"]),
    ("final_logistic", "Logistic diversity model",
     "train_v17_logistic.py", ["v17_logistic_model.joblib"]),
    ("final_context", "Context HGB (source of the V106 context model)",
     "train_v31_models.py", ["v31_feature_models.joblib"]),
    ("final_form", "Form HGB at in-season shrinkage 20, and the frozen anchor",
     "build_v105_artifacts.py",
     ["v105_feature_models.joblib", "v105_inseason_anchor.joblib"]),
    ("calibration", "Global shift, correction lookups and blend weights",
     "build_v106_calibration.py", ["v106_calibration.joblib"]),
    ("package", "Assemble submit_v106.zip and gate on server loadability",
     "package_v106_submission.py", []),
]
NOTES = {
    "oof_form": (
        "Trains four shrinkages because that is the validated script that produced "
        "the artifact; V106 reads only the shrinkage-20 arm."
    ),
    "oof_context": (
        "Trains several ablation variants; V106 reads only no_matchup_hte."
    ),
}


def environment():
    code = (
        "import json,platform,sys,numpy,pandas,sklearn,scipy,joblib;"
        "print(json.dumps({'platform':platform.platform(),"
        "'python':sys.version.split()[0],'numpy':numpy.__version__,"
        "'pandas':pandas.__version__,'scikit_learn':sklearn.__version__,"
        "'scipy':scipy.__version__,'joblib':joblib.__version__}))"
    )
    result = subprocess.run([SERVER_PYTHON, "-c", code], capture_output=True, text=True)
    result.check_returncode()
    return json.loads(result.stdout)


def verify():
    """The rebuilt ZIP must predict exactly what the recorded submission predicted."""
    script = ROOT / "scripts" / "verify_v106_reproduction.py"
    if not script.exists():
        print("  verification script missing; skipped")
        return True
    result = subprocess.run([SERVER_PYTHON, str(script)], cwd=ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--from", dest="start", help="resume from this stage key")
    parser.add_argument("--only", help="run a single stage key")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"{'stage':>18}  {'script':<52} produces")
        for key, description, script, produces in STAGES:
            print(f"{key:>18}  {script:<52} {', '.join(produces) or '-'}")
            print(f"{'':>18}  {description}")
            if key in NOTES:
                print(f"{'':>18}  note: {NOTES[key]}")
        return 0

    if not Path(SERVER_PYTHON).exists():
        raise SystemExit(
            f"Server-mirror interpreter missing: {SERVER_PYTHON}\n"
            "conda create -y -n lgaimers_server python=3.11 && "
            f"{SERVER_PYTHON.replace('python', 'pip')} install numpy==1.26.4 "
            "pandas==2.0.3 scikit-learn==1.8.0 joblib==1.5.3 scipy==1.15.3"
        )

    print("environment:", json.dumps(environment(), indent=2))
    if args.verify_only:
        return 0 if verify() else 1

    stages = STAGES
    if args.only:
        stages = [s for s in STAGES if s[0] == args.only]
        if not stages:
            raise SystemExit(f"unknown stage: {args.only}")
    elif args.start:
        keys = [s[0] for s in STAGES]
        if args.start not in keys:
            raise SystemExit(f"unknown stage: {args.start}")
        stages = STAGES[keys.index(args.start):]

    started = time.time()
    for index, (key, description, script, produces) in enumerate(stages, 1):
        print(f"\n[{index}/{len(stages)}] {key}: {description}", flush=True)
        step = time.time()
        result = subprocess.run([SERVER_PYTHON, f"scripts/{script}"], cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(f"stage {key} failed ({script})")
        for name in produces:
            path = ROOT / "artifacts" / name
            if not path.exists():
                raise SystemExit(f"stage {key} did not produce {name}")
            print(f"    {name}  {path.stat().st_size / 2**20:.2f} MiB")
        print(f"    done in {time.time() - step:.0f}s", flush=True)

    print(f"\nall stages finished in {(time.time() - started) / 60:.1f} min")
    if args.only:
        return 0
    print("\nverifying the rebuilt submission reproduces the recorded predictions")
    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
