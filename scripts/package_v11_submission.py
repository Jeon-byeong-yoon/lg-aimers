"""Package the V11 robust blend and calibration candidate."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    files = {
        Path("submission_src/v11_script.py"): "script.py",
        Path("submission_src/requirements.txt"): "requirements.txt",
        Path("scripts/feature_engineering_v2.py"): "feature_engineering_v2.py",
        Path("scripts/trackman_features.py"): "trackman_features.py",
        Path("scripts/target_encoding_v5.py"): "target_encoding_v5.py",
        Path("scripts/hierarchical_target_encoding_v6.py"): "hierarchical_target_encoding_v6.py",
        Path("artifacts/v6_ensemble.joblib"): "model/v6_ensemble.joblib",
    }
    output = Path("submissions/by_submitter/전병윤_RF_기준선/submit_v11_robust_blend.zip")
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for source, destination in files.items():
            archive.write(source, destination)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
