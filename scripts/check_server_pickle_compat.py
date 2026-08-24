"""Load every model artifact under the evaluation server's package versions.

The evaluation server runs numpy 1.26.4. Artifacts pickled under numpy 2.x embed
``numpy.random.Generator`` by class reference, which numpy 1.x cannot resolve and
which fails at load time with:

    ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module.

This cost one submission on V93. Run this script with the server-mirror
interpreter before every packaging step.
"""

import sys
import traceback
from pathlib import Path

import joblib


def main():
    targets = sys.argv[1:] or sorted(str(p) for p in Path("artifacts").glob("*.joblib"))
    failures = []
    for path in targets:
        try:
            joblib.load(path)
            print(f"  OK    {path}")
        except Exception as error:
            failures.append((path, error))
            print(f"  FAIL  {path}\n        {type(error).__name__}: {error}")
    print()
    if failures:
        print(f"{len(failures)} / {len(targets)} artifact(s) are NOT server-loadable")
        return 1
    print(f"all {len(targets)} artifact(s) load under this interpreter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
