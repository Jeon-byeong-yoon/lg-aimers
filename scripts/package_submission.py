"""Create the exact evaluation-server ZIP without local data or outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/ensemble.joblib")
    parser.add_argument("--output", default="submissions/ensemble_stable.zip")
    parser.add_argument("--variant", choices=("stable", "affine"), default="stable")
    args = parser.parse_args()
    files = {
        Path("submission_src/script.py"): "script.py",
        Path("submission_src/requirements.txt"): "requirements.txt",
        Path(args.model): "model/ensemble.joblib",
    }
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing submission inputs: {missing}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for source, destination in files.items():
            if destination == "script.py":
                script = source.read_text(encoding="utf-8")
                script = script.replace(
                    'os.environ.get("PREDICTION_VARIANT", "stable")',
                    f'os.environ.get("PREDICTION_VARIANT", "{args.variant}")',
                )
                archive.writestr(destination, script)
            else:
                archive.write(source, destination)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
