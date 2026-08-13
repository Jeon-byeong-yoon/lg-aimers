"""Package V2 directly into Jeon Byeong-yoon's submission workspace."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    files = {
        Path("submission_src/v2_script.py"): "script.py",
        Path("submission_src/requirements.txt"): "requirements.txt",
        Path("scripts/feature_engineering_v2.py"): "feature_engineering_v2.py",
        Path("artifacts/v2_ensemble.joblib"): "model/v2_ensemble.joblib",
    }
    output = Path("submissions/by_submitter/전병윤_RF_기준선/submit_v2.zip")
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for source, destination in files.items():
            archive.write(source, destination)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
