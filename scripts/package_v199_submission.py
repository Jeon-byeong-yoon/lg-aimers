"""Package V199: every component is now an average of many draws.

Every learned component is now an average: the network and the factorization
network over nine draws, Form and Context over six, CatBoost over four. The feature frame
is V175's 105 columns exactly; V187 lost 2.59 with the extra encoding groups, so they are
not here.

The archive grows to roughly 400 MB because four CatBoost models are about 108 MB each.
`docs/05` allows 10 GB compressed and 32 GB unpacked, so that is four percent of the
limit -- the earlier "220 MB is too big" worry was asserted without reading the rules and
was wrong.

Timing is the check that matters. Inference runs 34 model evaluations per row now, so the
full-frame run is re-measured against the 600-second cap rather than inherited.

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
        Path("artifacts/v199_calibration.joblib"): "model/v25_calibration.joblib",
        Path("artifacts/v193_networks.joblib"): "model/v193_networks.joblib",
        Path("artifacts/v196_factorizations.joblib"): "model/v196_factorizations.joblib",
        Path("artifacts/v199_catboost_meta.joblib"): "model/v199_catboost_meta.joblib",
        Path("artifacts/v199_catboosts/catboost_seed42.cbm"):
            "model/catboost_seed42.cbm",
        Path("artifacts/v199_catboosts/catboost_seed1004.cbm"):
            "model/catboost_seed1004.cbm",
        Path("artifacts/v199_catboosts/catboost_seed2024.cbm"):
            "model/catboost_seed2024.cbm",
        Path("artifacts/v199_catboosts/catboost_seed777.cbm"):
            "model/catboost_seed777.cbm",
        Path("artifacts/v105_inseason_anchor.joblib"): "model/v105_inseason_anchor.joblib",
        Path("artifacts/v154_inseason_anchor.joblib"): "model/v154_inseason_anchor.joblib",
        Path("scripts/inseason_prior_v153.py"): "inseason_prior_v153.py",
    }
    verify_server_loadable(
        [str(source) for source in files
         if str(source).endswith((".joblib", ".cbm"))]
    )
    output = Path("submissions/by_submitter/전병윤_RF_기준선/submit_v199.zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for source, destination in files.items():
            archive.write(source, destination)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
