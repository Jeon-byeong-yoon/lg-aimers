"""V131: re-rank every rejected candidate under a three-season instrument.

Three leaderboard results now say the 2024 point estimate is biased low, and the bias
grows as the estimate shrinks:

    submission          2024 estimate   actual   ratio
    V114 network              +24.5     +29.51   1.20
    V117 CatBoost             +9.9      +25.67   2.59
    V122 weights             +5.01      +18.80   3.75

V121's explanation for V117 -- that CatBoost's gains are heterogeneous across seasons,
so 2024 happened to be the low one -- cannot cover V122, which changed only weights.
A systematic bias that scales with the estimate is not three coincidences.

The mechanism is that all three folds answer the same question. Each predicts the
season immediately after its training window: 2022 follows 2019-2021, 2023 follows
2019-2022, 2024 follows 2019-2023, and 2025 will follow 2019-2024. They differ only in
history depth, which V121 measured as irrelevant (757/748/765/745 for one through four
seasons). So they are three equally valid measurements of the same quantity, and using
one while discarding two throws away two thirds of the evidence.

The instrument is therefore the *equally weighted* average of the three per-season
means, not a pooled bootstrap over all rows. That distinction is V95's hard-won lesson:
2023's gains run three to seven times the other seasons', so pooling by row lets 2023
dominate, and V95 was misled exactly that way -- a pooled bootstrap of +21.3 points
whose transfer-relevant segment was negative. Equal weighting gives 2023 one third,
no more.

    average    = (g2022 + g2023 + g2024) / 3
    SE(average) = sqrt(SE2022^2 + SE2023^2 + SE2024^2) / 3

with each SE recovered from that season's stored bootstrap interval.

What does *not* change is the monthly block win rate floor of 75%. That is a different
kind of evidence -- consistency across twenty monthly blocks rather than magnitude on
three seasons -- and nothing in the leaderboard record impugns it. Every candidate
resurrected here has to clear it on the same terms as before, which is what keeps this
a correction of a mis-specified instrument rather than a relaxation to taste.
"""

import json
import math
from pathlib import Path

P = 100000.0 / 0.25
BLOCK_FLOOR = 0.75
SOURCES = {
    "V116_catboost": "artifacts/v116_catboost_metrics.json",
    "V118_lightgbm": "artifacts/v118_lightgbm_metrics.json",
    "V120b_cb_weight": "artifacts/v120b_catboost_weight_metrics.json",
    "V123_capacity": "artifacts/v123_catboost_capacity_metrics.json",
    "V127_drift": "artifacts/v127_drift_refit_metrics.json",
    "V129_network": "artifacts/v129_network_retest_metrics.json",
    "V130_interaction": "artifacts/v130_interaction_network_metrics.json",
}
# V124 stores its candidates under two separate keys.
MIXTURE_SOURCES = {"V124_mixture": ("artifacts/v124_variant_mixture_metrics.json",
                                    ("mixtures", "slot_ladder"))}


def standard_error(entry):
    """Recover a season's SE from its stored 95% interval."""
    return (entry["ci95_high"] - entry["ci95_low"]) / 3.9199


def score(metrics):
    seasons = metrics.get("season_bootstrap")
    if not seasons or not all(str(y) in seasons for y in (2022, 2023, 2024)):
        return None
    means = [seasons[str(y)]["mean"] for y in (2022, 2023, 2024)]
    errors = [standard_error(seasons[str(y)]) for y in (2022, 2023, 2024)]
    average = sum(means) / 3.0
    combined = math.sqrt(sum(e * e for e in errors)) / 3.0
    return {
        "average_points": average * P,
        "se_points": combined * P,
        "ci95_low_points": (average - 1.96 * combined) * P,
        "season_points": [m * P for m in means],
        "blocks": metrics.get("monthly_block_win_rate"),
        "old_2024_ci_low_points": metrics["bootstrap_2024"]["ci95_low"] * P,
        "old_2024_mean_points": metrics["bootstrap_2024"]["mean"] * P,
    }


def main():
    rows = []
    for label, path in SOURCES.items():
        file = Path(path)
        if not file.exists():
            print(f"  missing {path}")
            continue
        payload = json.loads(file.read_text(encoding="utf-8"))
        for name, metrics in payload.get("results", {}).items():
            evaluated = score(metrics)
            if evaluated:
                rows.append((label, name, evaluated))
    for label, (path, keys) in MIXTURE_SOURCES.items():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in keys:
            for name, metrics in payload.get(key, {}).items():
                evaluated = score(metrics)
                if evaluated:
                    rows.append((label, f"{key}:{name}", evaluated))

    survivors = [r for r in rows
                 if r[2]["ci95_low_points"] > 0
                 and (r[2]["blocks"] or 0) >= BLOCK_FLOOR
                 and r[2]["old_2024_ci_low_points"] <= 0]
    print(f"{len(rows)} candidates re-scored; "
          f"{len(survivors)} were rejected by the old instrument and pass the new one\n")
    print(f"{'experiment':>18} {'candidate':>26} {'avg':>7} {'CIlo':>7} {'2022':>7} "
          f"{'2023':>8} {'2024':>7} {'blocks':>7} {'old CIlo':>9}")
    for label, name, e in sorted(rows, key=lambda r: -r[2]["ci95_low_points"])[:28]:
        mark = " *" if (label, name, e) in survivors else ""
        print(f"{label:>18} {name:>26} {e['average_points']:7.2f} "
              f"{e['ci95_low_points']:7.2f} {e['season_points'][0]:7.2f} "
              f"{e['season_points'][1]:8.2f} {e['season_points'][2]:7.2f} "
              f"{(e['blocks'] or 0):7.0%} {e['old_2024_ci_low_points']:9.2f}{mark}")

    output = Path("artifacts/v131_three_season_reranking.json")
    output.write_text(json.dumps({
        "experiment": "V131_three_season_reranking",
        "motivation": {
            "leaderboard": [
                {"version": "V114", "estimate_2024": 24.5, "actual": 29.51,
                 "ratio": 1.20},
                {"version": "V117", "estimate_2024": 9.9, "actual": 25.67,
                 "ratio": 2.59},
                {"version": "V122", "estimate_2024": 5.01, "actual": 18.80,
                 "ratio": 3.75},
            ],
            "argument": (
                "The bias grows as the estimate shrinks, and V122 changed only weights, "
                "so CatBoost's season heterogeneity cannot explain it. All three folds "
                "predict the season immediately after their training window, exactly as "
                "2025 will, and V121 showed history depth irrelevant, so they are three "
                "equally valid measurements rather than one plus two distractions."
            ),
        },
        "instrument": (
            "Equally weighted average of the three per-season means, with SE combined "
            "from each season's stored interval. Deliberately not a pooled bootstrap: "
            "2023's gains run three to seven times the others', so row pooling lets it "
            "dominate, which is how V95 was misled."
        ),
        "unchanged": (
            f"Monthly block win rate floor of {BLOCK_FLOOR:.0%}. Consistency across "
            "twenty blocks is a different kind of evidence and nothing in the "
            "leaderboard record impugns it."
        ),
        "candidates": [{"experiment": l, "candidate": n, **e} for l, n, e in rows],
        "resurrected": [{"experiment": l, "candidate": n, **e} for l, n, e in survivors],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {output}")
    if survivors:
        print("\nresurrected (rejected by 2024-only, pass three-season + blocks):")
        for label, name, e in sorted(survivors, key=lambda r: -r[2]["ci95_low_points"]):
            print(f"  {label} / {name}: average {e['average_points']:+.2f} "
                  f"CI low {e['ci95_low_points']:+.2f}, blocks {e['blocks']:.0%}")


if __name__ == "__main__":
    main()
