"""Package V105: V96 with the reconstruction smoothing refitted."""

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
        Path("submission_src/v105_script.py"): "script.py",
        Path("submission_src/requirements.txt"): "requirements.txt",
        Path("scripts/feature_engineering_v2.py"): "feature_engineering_v2.py",
        Path("scripts/trackman_features.py"): "trackman_features.py",
        Path("scripts/target_encoding_v5.py"): "target_encoding_v5.py",
        Path("scripts/hierarchical_target_encoding_v6.py"): "hierarchical_target_encoding_v6.py",
        Path("scripts/stable_form_features_v22.py"): "stable_form_features_v22.py",
        Path("scripts/contextual_trackman_v24.py"): "contextual_trackman_v24.py",
        Path("scripts/inseason_asof_features_v92.py"): "inseason_asof_features_v92.py",
        Path("artifacts/v6_ensemble.joblib"): "model/v6_ensemble.joblib",
        Path("artifacts/v17_logistic_model.joblib"): "model/v17_logistic_model.joblib",
        Path("artifacts/v105_feature_models.joblib"): "model/v105_feature_models.joblib",
        Path("artifacts/v105_calibration.joblib"): "model/v25_calibration.joblib",
        Path("artifacts/v105_inseason_anchor.joblib"): "model/v105_inseason_anchor.joblib",
    }
    verify_server_loadable(
        [str(source) for source in files if str(source).endswith(".joblib")]
    )
    output = Path("submissions/by_submitter/전병윤_RF_기준선/submit_v105.zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for source, destination in files.items():
            archive.write(source, destination)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
