"""Package V207: the shipped CatBoosts cut to 1,000 trees.

Everything is V199 except the four CatBoost models (shrunk from 1200 to 1000 trees --
exact prefixes, no retraining, controls in build_v207 demanded prediction equality to
0.0) and the calibration layer, refitted on V204's cached 1000-prefix fold predictions.

The inference script is v199_script.py unchanged: the shrunk models predict with all
their trees by default and the in-archive artifact names stay identical, so the only
things the server sees differently are the model files and the calibration table. The
archive shrinks by about a sixth because trees are the bulk of a `.cbm`.

The load gate still runs on every artifact including the `cbm`: an installation
failure does not count against the daily quota, but a failure after `script.py`
starts does.
"""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SERVER_PYTHON = "/opt/anaconda3/envs/lgaimers_server/bin/python"


def verify_server_loadable(model_files):
    """Refuse to package unless every model loads under the server's numpy.

    The evaluation server runs numpy 1.26.4. Artifacts saved from a numpy 2.x
    interpreter embed numpy.random.Generator by class reference and fail to load
    there, which cost one submission on the first V93 attempt. Local inference
    succeeding proves nothing about the server, so this gate is mandatory.
    """
    import subprocess
    if not Path(SERVER_PYTHON).exists():
        raise SystemExit(
            f"Server-mirror interpreter missing: {SERVER_PYTHON}\n"
            "Create it with: conda create -y -n lgaimers_server python=3.11 && "
            "pip install numpy==1.26.4 pandas==2.0.3 scikit-learn==1.8.0 "
            "joblib==1.5.3 scipy==1.15.3"
        )
    result = subprocess.run(
        [SERVER_PYTHON, "scripts/check_server_pickle_compat.py", *model_files],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        raise SystemExit(
            "Aborting: at least one artifact is not loadable under the server's "
            "package versions. Retrain/re-save it with SERVER_PYTHON."
        )


def main():
    files = {
        Path("submission_src/v199_script.py"): "script.py",
        Path("submission_src/requirements.txt"): "requirements.txt",
        Path("scripts/feature_engineering_v2.py"): "feature_engineering_v2.py",
        Path("scripts/trackman_features.py"): "trackman_features.py",
        Path("scripts/target_encoding_v5.py"): "target_encoding_v5.py",
        Path("scripts/hierarchical_target_encoding_v6.py"): "hierarchical_target_encoding_v6.py",
        Path("scripts/stable_form_features_v22.py"): "stable_form_features_v22.py",
        Path("scripts/contextual_trackman_v24.py"): "contextual_trackman_v24.py",
        Path("scripts/inseason_asof_features_v92.py"): "inseason_asof_features_v92.py",
        Path("scripts/embedding_network_v111.py"): "embedding_network_v111.py",
        Path("scripts/interaction_network_v130.py"): "interaction_network_v130.py",
        Path("artifacts/v6_ensemble.joblib"): "model/v6_ensemble.joblib",
        Path("artifacts/v17_logistic_model.joblib"): "model/v17_logistic_model.joblib",
        Path("artifacts/v196_feature_models.joblib"): "model/v196_feature_models.joblib",
        Path("artifacts/v207_calibration.joblib"): "model/v25_calibration.joblib",
        Path("artifacts/v193_networks.joblib"): "model/v193_networks.joblib",
        Path("artifacts/v196_factorizations.joblib"): "model/v196_factorizations.joblib",
        Path("artifacts/v207_catboost_meta.joblib"): "model/v199_catboost_meta.joblib",
        Path("artifacts/v207_catboosts/catboost_seed42.cbm"):
            "model/catboost_seed42.cbm",
        Path("artifacts/v207_catboosts/catboost_seed1004.cbm"):
            "model/catboost_seed1004.cbm",
        Path("artifacts/v207_catboosts/catboost_seed2024.cbm"):
            "model/catboost_seed2024.cbm",
        Path("artifacts/v207_catboosts/catboost_seed777.cbm"):
            "model/catboost_seed777.cbm",
        Path("artifacts/v105_inseason_anchor.joblib"): "model/v105_inseason_anchor.joblib",
        Path("artifacts/v154_inseason_anchor.joblib"): "model/v154_inseason_anchor.joblib",
        Path("scripts/inseason_prior_v153.py"): "inseason_prior_v153.py",
    }
    verify_server_loadable(
        [str(source) for source in files
         if str(source).endswith((".joblib", ".cbm"))]
    )
    output = Path("submissions/by_submitter/전병윤_RF_기준선/submit_v207.zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for source, destination in files.items():
            archive.write(source, destination)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
