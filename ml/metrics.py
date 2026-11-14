import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, roc_auc_score, roc_curve)


def fdr_symbols(q):
    q = np.asarray(q, dtype=float)
    return np.where(q < 0.01, "**", np.where(q < 0.10, "*", ""))


def performance(y, probability):
    prediction = np.asarray(probability) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    ratio = lambda a, b: a / b if b else np.nan
    return {"AUROC": roc_auc_score(y, probability), "AP": average_precision_score(y, probability),
            "Brier": brier_score_loss(y, probability), "log_loss": log_loss(y, probability, labels=[0, 1]),
            "accuracy": accuracy_score(y, prediction), "sensitivity": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp), "PPV": ratio(tp, tp + fp), "NPV": ratio(tn, tn + fn),
            "F1": f1_score(y, prediction, zero_division=0), "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def bootstrap(y, probabilities, n_boot, seed):
    y = np.asarray(y)
    p = np.column_stack(list(probabilities.values()))
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 101)
    estimates, curves = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        estimates.append([[roc_auc_score(y[idx], q[idx]), average_precision_score(y[idx], q[idx])]
                          for q in p.T])
        roc = []
        for q in p.T:
            fpr, tpr, _ = roc_curve(y[idx], q[idx])
            roc.append(np.interp(grid, fpr, tpr))
        curves.append(roc)
    return np.asarray(estimates), np.asarray(curves), grid


def summarize(predictions, folder, config):
    folder.mkdir(parents=True, exist_ok=True)
    summary, bands, differences = [], [], []
    for (cohort, subgroup, strategy, algorithm), block in predictions.groupby(
            ["cohort", "subgroup", "strategy", "algorithm"], sort=False):
        pivot = block.pivot(index="patient_id", columns="feature_set", values="probability")
        y = block.drop_duplicates("patient_id").set_index("patient_id").loc[pivot.index, "outcome"].to_numpy()
        if np.unique(y).size < 2:
            continue
        probabilities = {name: pivot[name].to_numpy() for name in pivot}
        boot, curves, grid = bootstrap(y, probabilities, config["bootstrap"], config["seed"])
        context = dict(cohort=cohort, subgroup=subgroup, strategy=strategy, algorithm=algorithm)
        for j, (name, p) in enumerate(probabilities.items()):
            row = dict(context, feature_set=name, n=len(y), **performance(y, p))
            for k, metric in enumerate(["AUROC", "AP"]):
                row[metric + "_low"], row[metric + "_high"] = np.percentile(boot[:, j, k], [2.5, 97.5])
            summary.append(row)
            low, high = np.percentile(curves[:, j], [2.5, 97.5], axis=0)
            bands.extend([dict(context, feature_set=name, FPR=x, low=l, high=h)
                          for x, l, h in zip(grid, low, high)])
        names = list(probabilities)
        comparisons = [("clinical_bai_microbiome", "clinical_bai"),
                       ("integrated", "microbiome"), ("integrated", "proteome")]
        for target, reference in comparisons:
            if target not in names or reference not in names:
                continue
            j, k = names.index(target), names.index(reference)
            for m, metric in enumerate(["AUROC", "AP"]):
                delta = boot[:, j, m] - boot[:, k, m]
                low, high = np.percentile(delta, [2.5, 97.5])
                score = roc_auc_score if m == 0 else average_precision_score
                differences.append(dict(context, target=target, reference=reference, metric=metric,
                                        difference=score(y, probabilities[target]) - score(y, probabilities[reference]),
                                        low=low, high=high))
    pd.DataFrame(summary).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(bands).to_csv(folder / "roc_bands.csv", index=False)
    pd.DataFrame(differences).to_csv(folder / "paired_differences.csv", index=False)
    return pd.DataFrame(summary)
