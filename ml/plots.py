import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from .explain import save_plot


def plot_results(folder):
    predictions = pd.read_csv(folder / "predictions.csv")
    metrics = pd.read_csv(folder / "metrics.csv")
    roc_bands = pd.read_csv(folder / "roc_bands.csv")
    output = folder / "figures"
    output.mkdir(exist_ok=True)
    for (cohort, subgroup, strategy), block in predictions.groupby(["cohort", "subgroup", "strategy"], sort=False):
        prefix = f"{cohort}_{subgroup}_{strategy}"
        for feature_set, frame in block.groupby("feature_set", sort=False):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for algorithm, d in frame.groupby("algorithm", sort=False):
                fpr, tpr, _ = roc_curve(d.outcome, d.probability)
                precision, recall, _ = precision_recall_curve(d.outcome, d.probability)
                axes[0].plot(fpr, tpr, label=f"{algorithm} ({roc_auc_score(d.outcome, d.probability):.3f})")
                axes[1].plot(recall, precision, label=f"{algorithm} ({average_precision_score(d.outcome, d.probability):.3f})")
            axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.7)
            axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"{feature_set}: AUROC")
            axes[1].set(xlabel="Recall", ylabel="Precision", title=f"{feature_set}: AP")
            for ax in axes:
                ax.legend(fontsize=7)
            save_plot(output / f"{prefix}_{feature_set}_algorithms")
            algorithms = frame.algorithm.unique()
            fig, axes = plt.subplots(2, int(np.ceil(len(algorithms) / 2)), figsize=(12, 6))
            for ax, algorithm in zip(np.atleast_1d(axes).ravel(), algorithms):
                d = frame[frame.algorithm == algorithm]
                cm = confusion_matrix(d.outcome, d.probability >= 0.5, labels=[0, 1])
                ConfusionMatrixDisplay(cm, display_labels=["BBD", "MBD"]).plot(ax=ax, colorbar=False)
                ax.set_title(algorithm)
            for ax in np.atleast_1d(axes).ravel()[len(algorithms):]:
                ax.axis("off")
            fig.tight_layout()
            save_plot(output / f"{prefix}_{feature_set}_confusion")
        selected = block[block.algorithm == "selected"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for feature_set, d in selected.groupby("feature_set", sort=False):
            fpr, tpr, _ = roc_curve(d.outcome, d.probability)
            precision, recall, _ = precision_recall_curve(d.outcome, d.probability)
            line, = axes[0].plot(fpr, tpr, label=f"{feature_set} ({roc_auc_score(d.outcome, d.probability):.3f})")
            band = roc_bands[(roc_bands.cohort == cohort) & (roc_bands.subgroup == subgroup) &
                             (roc_bands.strategy == strategy) & (roc_bands.algorithm == "selected") &
                             (roc_bands.feature_set == feature_set)]
            axes[0].fill_between(band.FPR, band.low, band.high, color=line.get_color(), alpha=0.12)
            axes[1].plot(recall, precision, label=f"{feature_set} ({average_precision_score(d.outcome, d.probability):.3f})")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="AUROC")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Average precision")
        for ax in axes:
            ax.legend(fontsize=7)
        save_plot(output / f"{prefix}_feature_sets")
        fig, ax = plt.subplots(figsize=(8, 4))
        groups = list(selected.groupby("feature_set", sort=False))
        positions, labels = [], []
        for i, (feature_set, d) in enumerate(groups):
            for outcome, offset, color in [(0, -0.17, "#2878B5"), (1, 0.17, "#C44E52")]:
                position = i + offset
                p = d.loc[d.outcome == outcome, "probability"]
                parts = ax.violinplot(p, positions=[position], widths=0.3, showmedians=True)
                for body in parts["bodies"]:
                    body.set_facecolor(color)
            positions.append(i)
            labels.append(feature_set)
        ax.set(xticks=positions, xticklabels=labels, ylabel="Predicted MBD probability")
        save_plot(output / f"{prefix}_probabilities")
        measures = ["AUROC", "AP", "accuracy", "sensitivity", "specificity", "PPV", "NPV", "F1"]
        angle = np.linspace(0, 2 * np.pi, len(measures), endpoint=False)
        closed_angle = np.r_[angle, angle[0]]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        selected_metrics = metrics[(metrics.cohort == cohort) & (metrics.subgroup == subgroup) &
                                   (metrics.strategy == strategy) & (metrics.algorithm == "selected")]
        for _, row in selected_metrics.iterrows():
            values = row[measures].to_numpy(float)
            ax.plot(closed_angle, np.r_[values, values[0]], label=row.feature_set)
        ax.set(xticks=angle, xticklabels=measures, ylim=(0, 1))
        ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.1), fontsize=7)
        save_plot(output / f"{prefix}_performance")
    for (cohort, subgroup, feature_set), frame in predictions[predictions.algorithm == "selected"].groupby(
            ["cohort", "subgroup", "feature_set"], sort=False):
        if frame.strategy.nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        for strategy, d in frame.groupby("strategy", sort=False):
            fpr, tpr, _ = roc_curve(d.outcome, d.probability)
            ax.plot(fpr, tpr, label=f"{strategy} ({roc_auc_score(d.outcome, d.probability):.3f})")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=feature_set)
        ax.legend()
        save_plot(output / f"{cohort}_{subgroup}_{feature_set}_full_vs_topk")
    differences = pd.read_csv(folder / "paired_differences.csv")
    if not differences.empty:
        d = differences[differences.algorithm == "selected"].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, max(3, len(d) * 0.3)))
        ax.hlines(np.arange(len(d)), d.low, d.high, color="#2878B5")
        ax.scatter(d.difference, np.arange(len(d)), color="#2878B5", s=20)
        ax.set_yticks(np.arange(len(d)), d.apply(
            lambda r: f"{r.cohort} | {r.subgroup} | {r.strategy} | {r.target} - {r.reference} | {r.metric}", axis=1), fontsize=7)
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("Difference (95% bootstrap CI)")
        save_plot(output / "incremental_performance")
