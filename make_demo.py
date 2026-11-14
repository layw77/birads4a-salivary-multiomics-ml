import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate(output, discovery=160, validation=120, seed=2026):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = discovery + validation
    latent = rng.normal(size=n)
    outcome = rng.binomial(1, 1 / (1 + np.exp(-latent)))
    ids = [f"SYNTHETIC_{i:04d}" for i in range(n)]
    bai = np.clip(np.rint(11 + 4 * outcome + rng.normal(0, 6, n)), 0, 63).astype(int)
    clinical = pd.DataFrame({
        "patient_id": ids, "outcome": outcome,
        "cohort": ["discovery"] * discovery + ["validation"] * validation,
        "bai": bai, "age": rng.normal(43 + 2 * outcome, 8, n),
        "bmi": rng.normal(23, 3, n), "diameter_mm": rng.lognormal(2.6 + 0.1 * outcome, 0.3, n),
        "menopause": rng.choice(["pre", "post"], n), "family_history": rng.choice(["no", "yes"], n),
        "smoking": rng.choice(["never", "former", "current"], n),
        "alcohol": rng.choice(["never", "former", "current"], n),
        "shape": rng.choice(["oval", "round", "irregular"], n),
        "orientation": rng.choice(["parallel", "nonparallel"], n),
        "margin": rng.choice(["circumscribed", "noncircumscribed"], n),
    })
    clinical.loc[rng.choice(n, max(1, n // 20), replace=False), "bmi"] = np.nan
    taxa = [f"Genus_{j:02d}" for j in range(1, 11)]
    abundances = np.exp(rng.normal(size=(n, 10)) + latent[:, None] * rng.normal(0, 0.5, (1, 10)))
    abundances /= abundances.sum(axis=1, keepdims=True)
    counts = pd.DataFrame([rng.multinomial(12000, row) for row in abundances], columns=taxa)
    counts.insert(0, "patient_id", ids)
    paired = (clinical.cohort == "validation") & (bai >= 8)
    proteins = [f"Protein_{j:02d}" for j in range(1, 9)]
    m = int(paired.sum())
    npx = rng.normal(5, 1, (m, 8)) + latent[paired.to_numpy(), None] * rng.normal(0, 0.5, (1, 8))
    lod = (npx < 2).astype(int)
    npx[rng.random(npx.shape) < 0.03] = np.nan
    npx = pd.DataFrame(npx, columns=proteins)
    npx.insert(0, "patient_id", clinical.loc[paired, "patient_id"].to_numpy())
    lod = pd.DataFrame(lod, columns=proteins)
    lod.insert(0, "patient_id", clinical.loc[paired, "patient_id"].to_numpy())
    for name, table in [("clinical", clinical), ("genus_counts", counts), ("proteomics", npx), ("protein_lod", lod)]:
        table.to_csv(output / f"{name}.csv", index=False)
    ko = pd.DataFrame(rng.lognormal(size=(m, 6)), columns=[f"K{j:05d}" for j in range(1, 7)])
    ko.insert(0, "patient_id", clinical.loc[paired, "patient_id"].to_numpy())
    ko.to_csv(output / "ko_abundances.csv", index=False)


def quick_config(config):
    config = json.loads(json.dumps(config))
    config.update(outer_folds=2, discovery_repeats=1, paired_repeats=1, inner_folds=2,
                  calibration_folds=2, bootstrap=30, shap_permutations=2, shap_background=10,
                  display_features=5, interaction_draws=8)
    config["parameters"] = {name: {key: [values[0]] for key, values in grid.items()}
                            for name, grid in config["parameters"].items()}
    for name in ["RF", "XGBoost", "LightGBM"]:
        config["parameters"][name]["n_estimators"] = [20]
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    generate(args.output, seed=args.seed)
    if args.quick:
        config = quick_config(json.loads(Path("config.json").read_text(encoding="utf-8")))
        (args.output / "quick_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
