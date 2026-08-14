"""Train V4: V2 ExtraTrees plus strictly prior-season Trackman HGB."""

from pathlib import Path

import joblib
import pandas as pd

from evaluate_v2 import extra_trees_pipeline, hist_gbdt_pipeline
from feature_engineering_v2 import add_row_features, select_v2_features
from trackman_features import add_trackman_features, build_lookup, prepare_trackman


TRACKMAN_COLUMNS = [
    "season", "game_month", "balls_before", "strikes_before", "pitcher_hand",
    "batter_hand", "pitch_type_group", "rel_speed", "spin_rate",
    "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side",
    "zone_speed",
]


def main() -> None:
    data = pd.read_csv("공모전 dataset/open/data/train.csv", encoding="utf-8-sig")
    y = data.pop("control_success").astype("uint8")
    data = data.drop(columns="row_id")
    prior = float(y.mean())
    base = select_v2_features(add_row_features(data, prior))
    trackman = prepare_trackman(
        pd.read_csv(
            "공모전 dataset/open/data/trackman_history.csv", usecols=TRACKMAN_COLUMNS
        )
    )
    with_trackman = add_trackman_features(base, trackman)

    extra, extra_columns = extra_trees_pipeline(base)
    trackman_hgb, trackman_columns = hist_gbdt_pipeline(with_trackman)
    extra.fit(base[extra_columns], y)
    trackman_hgb.fit(with_trackman[trackman_columns], y)

    bundle = {
        "models": {"extra_trees": extra, "trackman_hgb": trackman_hgb},
        "model_columns": {
            "extra_trees": extra_columns,
            "trackman_hgb": trackman_columns,
        },
        "weights": {"extra_trees": 0.40, "trackman_hgb": 0.60},
        "target_prior": prior,
        "raw_columns": list(data.columns),
        # Fixed official 2019--2024 history lookup for independent 2025 rows.
        "trackman_lookup_2025": build_lookup(trackman, 2025),
        "training_seasons": sorted(map(int, data["season"].unique())),
    }
    output = Path("artifacts/v4_ensemble.joblib")
    joblib.dump(bundle, output, compress=3)
    print(f"Saved {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
