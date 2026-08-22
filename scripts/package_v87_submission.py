"""V87 최종 3시드 제출 ZIP 생성."""

from pathlib import Path
from shutil import copyfile
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    files = {
        Path("submission_src/v87_script.py"): "script.py",
        Path("submission_src/requirements.txt"): "requirements.txt",
        Path("scripts/feature_engineering_v2.py"): "feature_engineering_v2.py",
        Path("scripts/trackman_features.py"): "trackman_features.py",
        Path("scripts/target_encoding_v5.py"): "target_encoding_v5.py",
        Path("scripts/hierarchical_target_encoding_v6.py"): "hierarchical_target_encoding_v6.py",
        Path("scripts/stable_form_features_v22.py"): "stable_form_features_v22.py",
        Path("scripts/contextual_trackman_v24.py"): "contextual_trackman_v24.py",
        Path("scripts/asof_features_v79.py"): "asof_features_v79.py",
        Path("artifacts/v6_ensemble.joblib"): "model/v6_ensemble.joblib",
        Path("artifacts/v17_logistic_model.joblib"): "model/v17_logistic_model.joblib",
        Path("artifacts/v38_feature_models.joblib"): "model/v25_feature_models.joblib",
        Path("artifacts/v41_calibration.joblib"): "model/v25_calibration.joblib",
        Path("artifacts/v87_final_models.joblib"): "model/v87_final_models.joblib",
    }
    missing = [str(source) for source in files if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package inputs: {missing}")
    output = Path("submissions/by_submitter/전병윤_RF_기준선/submit_v87.zip")
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for source, destination in files.items():
            archive.write(source, destination)
    upload = output.with_name("submit.zip")
    copyfile(output, upload)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")
    print(f"Copied byte-identical upload file to {upload}")


if __name__ == "__main__":
    main()
