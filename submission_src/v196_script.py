"""V161: RELIABILITY_SCALE 150 -> 300 for the three components that read it as a feature.

`ins_pitcher_reliability = inside_n / (inside_n + reliability_scale)` sets the half-trust
point in current-season pitches, and 150 was chosen a priori in V92. V155 found 150 optimal
for the *drift term*, which reads the reconstruction at shrinkage 3 and multiplies by the
column; Form, the network and the factorization network read it at shrinkage 20 as an
input, and V160 found 300 better there -- no fold losing, 2022 +0.54, 2023 +3.37,
2024 +0.77.

CatBoost keeps 150. It already takes its own reconstruction (V154's projected prior), so
only the existing career-prior block changes and no fourth reconstruction is needed.
"""

import math
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from contextual_trackman_v24 import CONTEXT, add_context_keys
from embedding_network_v111 import (
    EMBEDDING_SPECS, encode_categorical, encode_numeric, predict, restore,
)
from interaction_network_v130 import (
    predict as predict_factorization, restore as restore_factorization,
)
from inseason_asof_features_v92 import (
    add_inseason_features, drift_correction, feature_names,
)
from feature_engineering_v2 import add_row_features, select_v2_features
from hierarchical_target_encoding_v6 import add_test_hierarchical_encodings
from stable_form_features_v22 import add_stable_form_features
from target_encoding_v5 import add_test_target_encodings
from trackman_features import KEYS


TREE_WEIGHTS = {
    "extra_trees": 0.29483562599237795,
    "trackman_hgb": 0.2344456574665245,
    "encoded_hgb": 0.11617615437521091,
    "hierarchical_hgb": 0.35454256216588664,
}


def lookup_values(frame, lookup, columns, value="correction"):
    """Merge a segment lookup onto the rows, and refuse a total miss.

    An unmatched key yields 0, which is right for a 2025 debutant with no history
    but is also what a dtype mismatch produces -- and a mismatch would silently
    zero the whole term while the script kept emitting plausible numbers. The
    official five-row sample already misses on 40% of `pitcher_id` because 2025
    brings debutants, so partial misses are normal and only a *total* miss on a
    frame large enough for the reading to mean something is treated as the bug it
    would be.
    """
    keyed = frame[columns].copy()
    keyed["_order"] = np.arange(len(keyed))
    merged = keyed.merge(lookup, how="left", on=columns, validate="many_to_one")
    values = merged.sort_values("_order")[value]
    if len(lookup) > 1 and len(frame) >= 100 and values.notna().sum() == 0:
        raise ValueError(
            f"segment {columns} matched none of {len(frame)} rows against "
            f"{len(lookup)} lookup entries -- key dtypes have diverged"
        )
    return values.fillna(0).to_numpy()


def catboost_frame(features, raw, meta):
    """Rebuild the training frame, aborting only on the failure that is truly a bug.

    Every categorical is stringified, so a dtype difference between training and
    inference (an int column arriving as float, say) would turn *every* value into an
    unseen category. CatBoost tolerates unseen categories by design, so the component
    would keep emitting plausible numbers while carrying no identity signal at all.
    Such a mismatch drives coverage to exactly zero, which the floor below catches.

    Unseen values on their own are not a bug. The official five-row test sample
    already contains two `pitcher_id` values above the training maximum of 24633, so
    2025 brings genuine debutants; the chronological folds saw 13.8%-19.9% unseen
    pitchers and CatBoost gained anyway. A tight floor here would abort a perfectly
    good submission, which is the more expensive mistake, so a thin coverage on the
    closed-vocabulary state columns only warns.
    """
    work = features.copy()
    for name in meta["categorical"]:
        if name not in work:
            work[name] = raw[name]
    work = work[meta["columns"]].copy()
    for name in meta["categorical"]:
        work[name] = work[name].astype(str)
        known = set(meta["categorical_values"][name])
        covered = float(work[name].isin(known).mean())
        if covered < 0.02:
            raise ValueError(
                f"categorical '{name}' matches training values for only "
                f"{covered:.2%} of rows, which indicates a dtype or column mismatch "
                f"rather than new entities; {len(known)} known values, "
                f"example known={sorted(known)[:3]}, "
                f"example seen={work[name].unique()[:3].tolist()}"
            )
        if name in meta["closed_vocabulary"] and covered < 0.95:
            print(f"WARNING: closed-vocabulary column '{name}' covered {covered:.1%}")
    work[meta["numeric"]] = work[meta["numeric"]].astype(np.float32)
    return work


def attach_lookup(frame, lookup, columns):
    output = frame.copy()
    output["__original_index"] = output.index
    output = output.merge(lookup, how="left", on=columns, validate="many_to_one")
    output = output.set_index("__original_index")
    output.index.name = frame.index.name
    return output


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    submission = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    bundle = joblib.load("./model/v6_ensemble.joblib")
    logistic = joblib.load("./model/v17_logistic_model.joblib")
    feature_models = joblib.load("./model/v196_feature_models.joblib")
    calibration = joblib.load("./model/v25_calibration.joblib")
    inseason = joblib.load("./model/v105_inseason_anchor.joblib")
    inseason_projected = joblib.load("./model/v154_inseason_anchor.joblib")
    networks = joblib.load("./model/v193_networks.joblib")
    factorizations = joblib.load("./model/v196_factorizations.joblib")
    catboost_meta = joblib.load("./model/v154_catboost_meta.joblib")
    catboost = CatBoostClassifier()
    catboost.load_model("./model/v154_catboost.cbm")
    
    if test["row_id"].duplicated().any() or submission["row_id"].duplicated().any():
        raise ValueError("row_id must be unique")
    if set(test["row_id"]) != set(submission["row_id"]):
        raise ValueError("row_id sets differ")
        
    raw = test.drop(columns="row_id")[bundle["raw_columns"]]
    base = select_v2_features(add_row_features(raw, bundle["target_prior"]))
    encoded_raw = add_test_target_encodings(raw, bundle["target_encoding_lookups"])
    encoded_base = select_v2_features(add_row_features(encoded_raw, bundle["target_prior"]))
    hierarchical_raw = add_test_hierarchical_encodings(
        encoded_raw, bundle["hierarchical_lookups"]
    )
    hierarchical_base = select_v2_features(
        add_row_features(hierarchical_raw, bundle["target_prior"])
    )
    trackman_base = attach_lookup(base, bundle["trackman_lookup_2025"], KEYS)
    encoded_trackman = attach_lookup(
        encoded_base, bundle["trackman_lookup_2025"], KEYS
    )
    hierarchical_trackman = attach_lookup(
        hierarchical_base, bundle["trackman_lookup_2025"], KEYS
    )
    feature_sets = {
        "extra_trees": base,
        "trackman_hgb": trackman_base,
        "encoded_hgb": encoded_trackman,
        "hierarchical_hgb": hierarchical_trackman,
    }
    tree_prediction = sum(
        TREE_WEIGHTS[name]
        * model.predict_proba(feature_sets[name][bundle["model_columns"][name]])[:, 1]
        for name, model in bundle["models"].items()
    )
    logistic_prediction = logistic["model"].predict_proba(
        hierarchical_trackman[logistic["columns"]]
    )[:, 1]
    v17_prediction = 0.95 * tree_prediction + 0.05 * logistic_prediction

    # V92/V93: the official asof_* counters are career-cumulative and continue
    # across seasons, so subtracting the frozen end-of-2024 state recovers each
    # entity's 2025-only history. Only this row's own official features and the
    # frozen training-time lookup are used, so evaluation rows never interact.
    # V105 uses a different smoothing constant in each place. The Form model was
    # fitted on features shrunk at 20, which V102-V104 showed to be an interior
    # optimum (10 and 15 were worse). The drift term, which is the only component
    # that extrapolates the league level, prefers even less shrinkage at 10.
    inseason_block = add_inseason_features(
        raw,
        inseason["anchors"],
        inseason["priors"],
        current_season=inseason["target_season"],
        shrinkage=calibration["feature_shrinkage"],
        reliability_scale=calibration["feature_reliability_scale"],
    )[feature_names()]
    drift_frame = add_inseason_features(
        raw,
        inseason["anchors"],
        inseason["priors"],
        current_season=inseason["target_season"],
        shrinkage=calibration["drift_shrinkage"],
        reliability_scale=calibration["drift_reliability_scale"],
    )
    # V154: the same reconstruction at the same smoothing, but shrinking toward the
    # population's projected within-season rate instead of a career average. Only
    # CatBoost consumes it.
    projected_block = add_inseason_features(
        raw,
        inseason_projected["anchors"],
        inseason_projected["priors"],
        current_season=inseason_projected["target_season"],
        shrinkage=calibration["feature_shrinkage"],
        reliability_scale=calibration["catboost_reliability_scale"],
    )[feature_names()]

    form_raw = add_stable_form_features(hierarchical_raw)
    form_base = select_v2_features(add_row_features(form_raw, bundle["target_prior"]))
    form_features = attach_lookup(form_base, bundle["trackman_lookup_2025"], KEYS)
    form_features = pd.concat([form_features, inseason_block], axis=1)
    # V193: six Form draws, averaged, for the same reason the network is averaged.
    # HistGradientBoosting's `random_state` only picks the 200,000-row subsample that
    # sets the bin thresholds, so its lottery is narrow -- range 2.97 on the 2024 fold
    # against the network's 17.94 -- but it is consistently signed and Form carries the
    # largest weight in the blend at 0.32.
    form_matrix = form_features[feature_models["form_columns"]]
    form_prediction = np.mean(
        [model.predict_proba(form_matrix)[:, 1]
         for model in feature_models["form_models"]],
        axis=0,
    )

    context_keys = add_context_keys(hierarchical_trackman)
    context_features = attach_lookup(
        context_keys, feature_models["context_lookup_2025"], KEYS + CONTEXT
    )
    # V196: six Context draws, averaged. Context's seed range on the 2024 fold is 6.54,
    # wider than Form's 2.97 even though both are HistGradientBoosting -- it is the
    # weaker model of the two, so it has more room to wobble. Its averaging curve is the
    # cleanest of the four components measured: monotone on all three seasons.
    context_matrix = context_features[feature_models["context_columns"]]
    context_prediction = np.mean(
        [model.predict_proba(context_matrix)[:, 1]
         for model in feature_models["context_models"]],
        axis=0,
    )

    # V96 refitted weights. V93 replaced the Form model but kept V41's 0.55/0.32/0.13,
    # which was fitted when Form was the weaker V38 model; the same mis-specification
    # after V38 was worth +6.54 points when V41 refitted it.
    # V114: the entity-embedding network. Every other component sees a pitcher only
    # as scalar summaries; this one holds a learned vector per pitcher, batter and
    # team, so identity interacts with the count and the runners directly. It reuses
    # the Form feature frame, minus `season`: passing an out-of-range season to a
    # network invites unbounded extrapolation that the fixed calibration shift could
    # not absorb, and dropping it was measurably safe.
    categorical_columns = [column for column, _ in EMBEDDING_SPECS]
    network_frame = form_features.copy()
    for column in categorical_columns:
        if column not in network_frame:
            network_frame[column] = raw[column]
    # V189: three independent draws, averaged. V164 rejected this on readings taken
    # against seed 42, which V188 showed to be the best of five draws on the 2024 fold
    # and the *worst* of five on 2022 -- seed luck is fold-specific noise, and a
    # deployed single draw is one more of them. Against each of the five single-seed
    # baselines in turn the average gains every season, and four of the five lose 2024
    # to it. The three bundles share vocabulary and statistics by construction, but each
    # carries its own so a future edit cannot pair one seed's weights with another's
    # encoding.
    network_prediction = np.mean(
        [
            predict(
                restore(bundle),
                encode_categorical(network_frame, bundle["vocabularies"]),
                encode_numeric(
                    network_frame[bundle["numeric_columns"]].to_numpy(dtype=float),
                    bundle["numeric_statistics"],
                ),
            )
            for bundle in networks
        ],
        axis=0,
    )

    # V117: CatBoost over the Form feature set with `te_*`/`hte_*` removed, so its
    # own ordered target statistics replace our hand-built encodings rather than
    # consuming them. V116 found that variant better than the full set at every
    # weight and both configs, which is V94b's league-contamination finding arriving
    # from the other direction. `season` is kept: for a tree, 2025 falls in the
    # terminal bin and is scored as 2024 was, and every validation fold predicted a
    # season later than any it trained on, so the measured behaviour transfers.
    # Same frame as Form's, with the in-season columns swapped for the projected-prior
    # ones. Everything else is identical, so the swap is the only difference.
    catboost_source = pd.concat(
        [form_features.drop(columns=list(projected_block.columns)), projected_block],
        axis=1,
    )
    catboost_prediction = catboost.predict_proba(
        catboost_frame(catboost_source, raw, catboost_meta)
    )[:, 1]

    # V138: a factorization network in part of the Context slot. Every other
    # component reaches an interaction implicitly -- trees by stacking splits, the
    # V111 network by asking a ReLU stack to imitate a product -- while this one
    # computes the pairwise field products directly. It was rejected as a sixth
    # component on standalone skill, but the Context slot rewards independence rather
    # than skill: at latent 8 it is the weakest of the sizes tried and the least
    # correlated with the rest (0.710 on 2024 against Context's 0.817), and it is the
    # only one that keeps 2024 positive. Latent 6 scores better standalone on every
    # season and fails. It reuses the network's frame, which carries the same 93
    # numeric columns, with its own vocabularies and normalisation.
    # V196: nine factorization draws, averaged. V188 rejected this on an estimator that
    # averaged a set containing seed 42 against baselines that also contained it; V191
    # showed that same bias reversing the network's curve, and corrected, this component
    # gains on every season at k = 4, 5, 7 and 8.
    factorization_prediction = np.mean(
        [
            predict_factorization(
                restore_factorization(bundle),
                encode_categorical(network_frame, bundle["vocabularies"]),
                encode_numeric(
                    network_frame[bundle["numeric_columns"]].to_numpy(dtype=float),
                    bundle["numeric_statistics"],
                ),
            )
            for bundle in factorizations
        ],
        axis=0,
    )

    weights = calibration["blend_weights"]
    prediction = (
        weights["v17"] * v17_prediction
        + weights["form"] * form_prediction
        + weights["context"] * context_prediction
        + weights["network"] * network_prediction
        + weights["catboost"] * catboost_prediction
        + weights["factorization"] * factorization_prediction
    )
    prediction += calibration["global_shift"]
    # Added only now, after every feature has been built, so it cannot reach a model.
    # A per-row function of one official column, so no evaluation row touches another.
    raw = raw.copy()
    raw["experience_bin"] = pd.cut(
        raw["asof_pitcher_n"],
        calibration["experience_edges"],
        labels=calibration["experience_labels"],
    ).astype(str)
    # V175: the platoon segment is resolved by two-strike state. The dtype is pinned
    # because the lookup is keyed on it and a silent int32/int64 divergence would
    # match nothing; `lookup_values` now refuses that outcome outright.
    raw["two_strike"] = (raw["strikes_before"] == 2).astype("int64")
    for correction in calibration["corrections"]:
        prediction += correction["weight"] * lookup_values(
            raw, correction["lookup"], correction["columns"]
        )
    prediction = np.clip(prediction, 0, 1)

    # The V92 post-hoc drift term stays: it is complementary to the in-model
    # features, not redundant (dropping it cut the monthly block win rate from
    # 100% to 70%). Its weight was refitted alongside the blend weights.
    prediction = np.clip(
        prediction + drift_correction(drift_frame, calibration["drift_weight"]), 0, 1
    )

    if len(prediction) != len(test) or any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1
        for value in prediction
    ):
        raise ValueError("Invalid prediction vector")
    submission["control_success"] = submission["row_id"].map(
        dict(zip(test["row_id"], prediction))
    )
    os.makedirs("./output", exist_ok=True)
    submission.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved rows={len(submission)}, prediction_mean={prediction.mean():.6f}")


if __name__ == "__main__":
    main()
