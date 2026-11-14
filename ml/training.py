import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from .metrics import summarize
from .models import make_model, underlying
from .preprocessing import model_importance
from .validation import validate_study


def choose(records):
    return max(records, key=lambda row: row["mean_AUROC"])


def tune(X, y, spec, config, calibrated, top_k=None):
    cv = list(StratifiedKFold(config["inner_folds"], shuffle=True,
                              random_state=config["seed"]).split(X, y))
    records = []
    for name in config["algorithms"]:
        for params in ParameterGrid(config["parameters"][name]):
            scores, briers, losses, counts = [], [], [], []
            for train, test in cv:
                fitted = make_model(name, params, spec, config, calibrated, top_k).fit(X.iloc[train], y[train])
                probability = fitted.predict_proba(X.iloc[test])[:, 1]
                scores.append(roc_auc_score(y[test], probability))
                briers.append(brier_score_loss(y[test], probability))
                losses.append(log_loss(y[test], probability, labels=[0, 1]))
                counts.append(len(underlying(fitted)["select"].get_feature_names_out()))
            records.append(dict(algorithm=name, params=params, mean_AUROC=float(np.mean(scores)),
                                mean_Brier=float(np.mean(briers)), mean_log_loss=float(np.mean(losses)),
                                n_features=float(np.mean(counts))))
    winners = {name: choose([r for r in records if r["algorithm"] == name])
               for name in config["algorithms"]}
    return choose(records), winners, records


def fit_record(record, X, y, spec, config, calibrated, top_k):
    return make_model(record["algorithm"], record["params"], spec, config,
                      calibrated, top_k).fit(X, y)


def append_predictions(rows, X, y, probabilities, feature_set, strategy, cohort, repeat, fold, subgroup=True):
    for algorithm, p in probabilities.items():
        frame = pd.DataFrame({"patient_id": X.index, "outcome": y, "probability": p,
                              "algorithm": algorithm, "feature_set": feature_set, "strategy": strategy,
                              "cohort": cohort, "repeat": repeat, "fold": fold, "subgroup": "overall"})
        rows.append(frame)
        if subgroup:
            rows.append(frame.loc[X.bai.to_numpy() >= 8].assign(subgroup="anxiety_positive"))


def record_search(records, chosen, context):
    return [context | row | dict(params=json.dumps(row["params"], sort_keys=True),
                                  selected=(row == chosen)) for row in records]


def train(X, external, specs, config, folder, paired=False):
    report = validate_study(X, external, specs, config, paired, include_explanation=False)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (folder / "models").mkdir(exist_ok=True)
    y = X.outcome.to_numpy(int)
    calibrated = not paired
    repeats = config["paired_repeats"] if paired else config["discovery_repeats"]
    cohort = "paired_oof" if paired else "discovery_oof"
    rows, searches, fold_rows, importances, selected_features = [], [], [], [], []
    final_models = {}
    for feature_set, spec in specs.items():
        strategies = [("full", None)]
        if paired:
            strategies += [(f"top{k}", k) for k in config["top_k"]]
        for strategy, top_k in strategies:
            for repeat in range(1, repeats + 1):
                cv = StratifiedKFold(config["outer_folds"], shuffle=True, random_state=config["seed"] + repeat - 1)
                for fold, (tr, te) in enumerate(cv.split(X, y), 1):
                    training, test = X.iloc[tr], X.iloc[te]
                    chosen, winners, records = tune(training, y[tr], spec, config, calibrated, top_k)
                    context = dict(feature_set=feature_set, strategy=strategy, repeat=repeat, fold=fold, stage="outer")
                    searches.extend(record_search(records, chosen, context))
                    fitted = {name: fit_record(record, training, y[tr], spec, config, calibrated, top_k)
                              for name, record in winners.items()}
                    name = chosen["algorithm"]
                    selected = fitted[name] if chosen == winners[name] else fit_record(
                        chosen, training, y[tr], spec, config, calibrated, top_k)
                    probabilities = {name: model.predict_proba(test)[:, 1] for name, model in fitted.items()}
                    probabilities["selected"] = selected.predict_proba(test)[:, 1]
                    append_predictions(rows, test, y[te], probabilities, feature_set, strategy, cohort, repeat, fold,
                                       subgroup=not paired)
                    fold_rows.extend([dict(context, patient_id=patient) for patient in test.index])
                    base = underlying(selected)
                    transformed = base[:-1].transform(training)
                    importance = model_importance(base["classifier"], transformed, y[tr], config["seed"])
                    names = base["select"].get_feature_names_out()
                    importances.extend([dict(context, algorithm=name, feature=f, importance=v)
                                        for f, v in zip(names, importance)])
                    selected_features.extend([dict(context, algorithm=name, feature=f) for f in names])
                    print(f"{feature_set} {strategy}: repeat {repeat}, fold {fold}, selected {name}", flush=True)
            chosen, winners, records = tune(X, y, spec, config, calibrated, top_k)
            context = dict(feature_set=feature_set, strategy=strategy, repeat=0, fold=0, stage="final")
            searches.extend(record_search(records, chosen, context))
            fitted = {name: fit_record(record, X, y, spec, config, calibrated, top_k)
                      for name, record in winners.items()}
            selected = fitted[chosen["algorithm"]] if chosen == winners[chosen["algorithm"]] else fit_record(
                chosen, X, y, spec, config, calibrated, top_k)
            fitted["selected"] = selected
            for name, model in fitted.items():
                joblib.dump(model, folder / "models" / f"{feature_set}_{strategy}_{name}.joblib", compress=3)
            final_models[f"{feature_set}_{strategy}"] = dict(
                algorithm=chosen["algorithm"], params=chosen["params"],
                features=underlying(selected)["select"].get_feature_names_out().tolist(),
                calibrated=calibrated or chosen["algorithm"] == "SVM")
            if external is not None:
                probabilities = {name: model.predict_proba(external)[:, 1] for name, model in fitted.items()}
                append_predictions(rows, external, external.outcome.to_numpy(int), probabilities, feature_set,
                                   strategy, "validation", 0, 0)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(folder / "fold_predictions.csv", index=False)
    keys = ["patient_id", "outcome", "algorithm", "feature_set", "strategy", "cohort", "subgroup"]
    predictions = raw.groupby(keys, as_index=False, sort=False).probability.mean()
    predictions.to_csv(folder / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(folder / "folds.csv", index=False)
    pd.DataFrame(searches).to_csv(folder / "model_search.csv", index=False)
    pd.DataFrame(selected_features).to_csv(folder / "selected_features.csv", index=False)
    importance_frame = pd.DataFrame(importances)
    importance_frame.to_csv(folder / "fold_importance.csv", index=False)
    names = importance_frame.feature.unique()
    averaged = []
    for (feature_set, strategy), block in importance_frame.groupby(["feature_set", "strategy"], sort=False):
        table = block.pivot_table(index=["repeat", "fold"], columns="feature", values="importance", fill_value=0)
        table = table.reindex(columns=[n for n in names if n in table], fill_value=0)
        weights = table.clip(lower=0)
        weights = weights.div(weights.sum(axis=1).replace(0, 1), axis=0)
        averaged.extend([dict(feature_set=feature_set, strategy=strategy, feature=f, mean_importance=v)
                         for f, v in weights.mean().sort_values(ascending=False).items()])
    pd.DataFrame(averaged).to_csv(folder / "importance.csv", index=False)
    (folder / "selected_models.json").write_text(json.dumps(final_models, indent=2), encoding="utf-8")
    summarize(predictions, folder, config)
    return predictions
