import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def class_counts(y, context, minimum=1):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    if counts.min() < minimum:
        raise ValueError(f"{context}: BBD={counts[0]}, MBD={counts[1]}; each class requires at least {minimum}")
    return {"BBD": int(counts[0]), "MBD": int(counts[1])}


def required_features(spec):
    proteins = spec["proteins"]
    return (spec["numeric"] + spec["categorical"] + spec["microbes"] + proteins +
            [name.replace("protein__", "lod__", 1) for name in proteins])


def check_features(training, evaluation, spec, context):
    for role, frame in [("training", training), ("evaluation", evaluation)]:
        if frame is None:
            continue
        missing = set(required_features(spec)) - set(frame.columns)
        if missing:
            raise ValueError(f"{context} {role}: missing features {sorted(missing)}")
        for prefix in ["micro__", "protein__", "lod__"]:
            expected = {name for name in training.columns if name.startswith(prefix)}
            observed = {name for name in frame.columns if name.startswith(prefix)}
            if observed != expected:
                raise ValueError(f"{context} {role}: {prefix} feature schema differs from training")
        numeric = spec["numeric"] + spec["microbes"] + spec["proteins"]
        values = frame[numeric].to_numpy(float)
        if np.isinf(values).any():
            raise ValueError(f"{context} {role}: infinite numeric measurements")
        if spec["microbes"]:
            counts = frame[spec["microbes"]].to_numpy(float)
            if not np.isfinite(counts).all() or (counts < 0).any() or (counts.sum(axis=1) <= 0).any():
                raise ValueError(f"{context} {role}: invalid microbial counts")
    for name in spec["numeric"] + spec["categorical"]:
        if training[name].isna().all():
            raise ValueError(f"{context}: clinical feature {name} is entirely missing in training")
    if spec["proteins"]:
        lod = [name.replace("protein__", "lod__", 1) for name in spec["proteins"]]
        if not training[lod].isin([0, 1]).all().all():
            raise ValueError(f"{context}: missing or invalid protein LOD flags")


def validate_study(X, external, specs, config, paired=False, include_training=True, include_explanation=True):
    calibration = not paired or "SVM" in config["algorithms"]
    for key in ["outer_folds", "inner_folds", "calibration_folds"]:
        if not isinstance(config[key], int) or config[key] < 2:
            raise ValueError(f"{key} must be an integer of at least 2")
    for k in config["top_k"]:
        if not isinstance(k, int) or k < 1:
            raise ValueError("top_k must contain positive integers")
    report = {"development": class_counts(X.outcome, "development"), "partitions": []}
    if not paired:
        report["development_anxiety_positive"] = class_counts(
            X.loc[X.bai >= 8, "outcome"], "anxiety-positive development")
    if external is not None:
        if not X.index.intersection(external.index).empty:
            raise ValueError("Development and validation participant IDs must be disjoint")
        report["validation"] = class_counts(external.outcome, "temporal validation")
        report["validation_anxiety_positive"] = class_counts(
            external.loc[external.bai >= 8, "outcome"], "anxiety-positive validation")
    for name, spec in specs.items():
        check_features(X, external, spec, name)

    def check_partition(frame, context):
        for name, spec in specs.items():
            check_features(frame, None, spec, context + " " + name)
            if spec["proteins"]:
                proteins = spec["proteins"]
                lod = [name.replace("protein__", "lod__", 1) for name in proteins]
                available = ((frame[lod].mean().to_numpy() < config["protein_lod_fraction"]) &
                             frame[proteins].notna().any().to_numpy())
                if not available.any():
                    raise ValueError(f"{context} {name}: no proteins with observed training values pass LOD QC")

    def check_fit(frame, context):
        counts = class_counts(frame.outcome, context, config["calibration_folds"] if calibration else 1)
        report["partitions"].append(dict(partition=context, **counts))
        check_partition(frame, context)
        if calibration:
            folds = StratifiedKFold(config["calibration_folds"], shuffle=True, random_state=config["seed"] + 19)
            for fold, (tr, te) in enumerate(folds.split(frame, frame.outcome), 1):
                label = f"{context} calibration {fold}"
                counts = class_counts(frame.iloc[tr].outcome, label + " fit")
                report["partitions"].append(dict(partition=label + " fit", **counts))
                class_counts(frame.iloc[te].outcome, label + " evaluation")
                check_partition(frame.iloc[tr], label)

    def check_tuning(frame, context):
        class_counts(frame.outcome, context + " inner CV", config["inner_folds"])
        folds = StratifiedKFold(config["inner_folds"], shuffle=True, random_state=config["seed"])
        check_fit(frame, context + " refit")
        for fold, (tr, te) in enumerate(folds.split(frame, frame.outcome), 1):
            class_counts(frame.iloc[te].outcome, f"{context} inner {fold} evaluation")
            check_fit(frame.iloc[tr], f"{context} inner {fold} fit")

    if include_training:
        repeats = config["paired_repeats"] if paired else config["discovery_repeats"]
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("CV repeats must be a positive integer")
        class_counts(X.outcome, "outer CV", config["outer_folds"])
        check_tuning(X, "final model")
        for repeat in range(repeats):
            folds = StratifiedKFold(config["outer_folds"], shuffle=True, random_state=config["seed"] + repeat)
            for fold, (tr, te) in enumerate(folds.split(X, X.outcome), 1):
                context = f"repeat {repeat + 1} outer {fold}"
                class_counts(X.iloc[te].outcome, context + " evaluation")
                check_tuning(X.iloc[tr], context)
    if include_explanation:
        try:
            tr, te = train_test_split(np.arange(len(X)), test_size=config["shap_test_fraction"],
                                     stratify=X.outcome, random_state=config["seed"])
        except ValueError as error:
            raise ValueError(f"SHAP 80:20 split: {error}") from error
        report["explanation_holdout"] = class_counts(X.iloc[te].outcome, "SHAP holdout")
        check_tuning(X.iloc[tr], "SHAP training")
    return report


def input_summary(data, proteins):
    rows = []
    for cohort, block in data.groupby("cohort", sort=False):
        for subgroup, selected in [("overall", block), ("anxiety_positive", block.loc[block.bai >= 8]),
                                    ("paired_anxiety_positive", block.loc[(block.bai >= 8) & block.paired])]:
            rows.append(dict(cohort=cohort, subgroup=subgroup, n=len(selected),
                             BBD=int(selected.outcome.eq(0).sum()), MBD=int(selected.outcome.eq(1).sum())))
    missingness = []
    for cohort, block in data.loc[data.paired].groupby("cohort", sort=False):
        for protein in proteins:
            missingness.append(dict(cohort=cohort, protein=protein, n=len(block),
                                    missing=int(block[protein].isna().sum()), missing_rate=float(block[protein].isna().mean()),
                                    below_lod_rate=float(block[protein.replace("protein__", "lod__", 1)].mean())))
    return pd.DataFrame(rows), pd.DataFrame(missingness, columns=[
        "cohort", "protein", "n", "missing", "missing_rate", "below_lod_rate"])
