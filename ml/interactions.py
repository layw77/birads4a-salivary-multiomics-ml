import itertools
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator


def pairwise_shap(predict, background, samples, pair, draws, seed):
    """Estimate one off-diagonal Shapley interaction, including its one-half factor."""
    background, samples = np.asarray(background, float), np.asarray(samples, float)
    dimension = samples.shape[1]
    i, j = pair
    others = np.array([k for k in range(dimension) if k not in pair], dtype=int)
    if draws < 1 or i == j or min(pair) < 0 or max(pair) >= dimension:
        raise ValueError("Interaction requires distinct feature indices and positive draws")
    rng = np.random.default_rng(seed)
    exact = len(others) < draws.bit_length() and 2 ** len(others) <= draws
    coalitions, weights = [], []
    if exact:
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                coalitions.append(list(subset))
                weights.append(1 / (2 * (dimension - 1) * math.comb(dimension - 2, size)))
    else:
        for _ in range(draws):
            size = rng.integers(len(others) + 1)
            coalitions.append(rng.choice(others, size=size, replace=False))
        weights = np.full(draws, 0.5 / draws)
    masks = np.zeros((len(coalitions), 4, dimension), dtype=bool)
    for k, subset in enumerate(coalitions):
        masks[k, :, np.asarray(subset, dtype=int)] = True
    masks[:, 1, i] = True
    masks[:, 2, j] = True
    masks[:, 3, [i, j]] = True
    values, errors = [], []
    for sample in samples:
        masked = np.where(masks[:, :, None, :], sample, background[None, None, :, :])
        flattened = masked.reshape(-1, dimension)
        predictions = np.concatenate([np.asarray(predict(flattened[start:start + 4096]))
                                      for start in range(0, len(flattened), 4096)])
        response = predictions.reshape(len(coalitions), 4, len(background)).mean(axis=2)
        delta = response[:, 3] - response[:, 1] - response[:, 2] + response[:, 0]
        values.append(np.dot(weights, delta))
        errors.append(0.0 if exact else 0.5 * delta.std(ddof=1) / np.sqrt(draws) if draws > 1 else np.nan)
    return np.asarray(values), np.asarray(errors), "exact" if exact else "Monte Carlo"


def interaction_surface(coordinates, interaction, names, folder, smoothing=1.0):
    coordinates, interaction = np.asarray(coordinates, float), np.asarray(interaction, float)
    observed = np.isfinite(coordinates).all(axis=1) & np.isfinite(interaction)
    coordinates, interaction = coordinates[observed], interaction[observed]
    if len(coordinates) < 6:
        return {"surface": "insufficient_coordinates"}
    center, scale = coordinates.mean(axis=0), coordinates.std(axis=0)
    if (scale == 0).any():
        return {"surface": "constant_coordinate"}
    standardized = (coordinates - center) / scale
    a, b = standardized.T
    polynomial = np.column_stack([np.ones(len(a)), a, b, a * a, a * b, b * b])
    if np.linalg.matrix_rank(polynomial) < 6:
        return {"surface": "rank_deficient_quadratic"}
    smoother = RBFInterpolator(standardized, interaction, smoothing=smoothing, kernel="quintic", degree=2)
    x, y = np.meshgrid(np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 60),
                       np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), 60))
    grid = np.column_stack([x.ravel(), y.ravel()])
    fitted = smoother((grid - center) / scale).reshape(x.shape)
    pd.DataFrame({names[0]: x.ravel(), names[1]: y.ravel(), "SHAP_interaction": fitted.ravel()}).to_csv(
        folder / "shap_interaction_surface.csv", index=False)
    statistics = dict(zip(["minimum", "Q1", "median", "Q3", "maximum"],
                          np.quantile(fitted, [0, 0.25, 0.5, 0.75, 1]).tolist()))
    metadata = dict(surface="fitted", kernel="quintic", degree=2, smoothing=smoothing,
                    coordinates="standardized CLR and NPX", grid_shape=list(x.shape),
                    color_limits=[-0.060, 0.060], **statistics)
    (folder / "shap_interaction_surface.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(x, y, fitted, levels=np.linspace(-0.060, 0.060, 21),
                          cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.060, vcenter=0, vmax=0.060), extend="both")
    if fitted.max() > fitted.min():
        lines = ax.contour(x, y, fitted, levels=7, colors="black", linewidths=0.4, linestyles="dashed")
        ax.clabel(lines, inline=True, fontsize=6, fmt="%.3f")
    ax.scatter(*coordinates.T, s=10, color="black", alpha=0.4)
    ax.set(xlabel=names[0], ylabel=names[1])
    ax.text(0.02, 0.98, "\n".join(f"{key}: {value:.4f}" for key, value in statistics.items()),
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.colorbar(contour, ax=ax, label="Pairwise SHAP interaction (MBD probability)")
    fig.savefig(folder / "shap_interaction_surface.pdf", bbox_inches="tight")
    fig.savefig(folder / "shap_interaction_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return metadata
