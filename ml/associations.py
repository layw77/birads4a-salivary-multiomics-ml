import json

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, t

from .data import read_matrix
from .metrics import fdr_symbols


def partial_spearman(x, y, covariates, numeric, categorical):
    frame = covariates[numeric + categorical].copy()
    frame["__x"], frame["__y"] = np.asarray(x, float), np.asarray(y, float)
    frame = frame.dropna()
    n = len(frame)
    if n < 3:
        return dict(n=n, rho=np.nan, P=np.nan, df=0)
    ranks = frame[["__x", "__y"] + numeric].rank(method="average")
    indicators = (pd.get_dummies(frame[categorical].astype("category"), drop_first=True, dtype=float)
                  if categorical else pd.DataFrame(index=frame.index))
    design = np.column_stack([np.ones(n), ranks[numeric].to_numpy(), indicators.to_numpy()])
    rank = np.linalg.matrix_rank(design)
    degrees = n - rank - 1
    residual = ranks[["__x", "__y"]].to_numpy()
    residual = residual - design @ np.linalg.lstsq(design, residual, rcond=None)[0]
    norm = np.linalg.norm(residual, axis=0)
    if degrees < 1 or (norm < 1e-12).any():
        return dict(n=n, rho=np.nan, P=np.nan, df=int(degrees))
    rho = float(np.clip(np.dot(residual[:, 0], residual[:, 1]) / np.prod(norm), -1, 1))
    statistic = abs(rho) * np.sqrt(degrees / max(1 - rho * rho, np.finfo(float).eps))
    return dict(n=n, rho=rho, P=float(2 * t.sf(statistic, degrees)), df=int(degrees))


def ko_protein_associations(data, proteins, directory, config, folder):
    ko = read_matrix(directory / "ko_abundances.csv").apply(pd.to_numeric, errors="raise")
    values = ko.to_numpy(float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Predicted KO abundances must be nonnegative and finite or missing")
    if not ko.index.isin(data.index).all():
        raise ValueError("KO participants must be present in clinical.csv")
    selected = data.loc[(data.cohort == "validation") & data.paired & (data.bai >= 8) & (data.outcome == 1)]
    if selected.empty or not proteins:
        raise ValueError("KO associations require paired anxiety-positive MBD participants with proteomics")
    if not selected.index.isin(ko.index).all():
        raise ValueError("KO abundances are required for every paired anxiety-positive MBD participant")
    numeric, categorical = config["association_numeric"], config["association_categorical"]
    missing = set(numeric + categorical) - set(selected.columns)
    if missing:
        raise ValueError(f"Missing association covariates: {sorted(missing)}")
    for name in numeric:
        values = pd.to_numeric(selected[name], errors="raise").dropna()
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite association covariate: {name}")
    paired = data.loc[(data.cohort == "validation") & data.paired & (data.bai >= 8)]
    retained = [name for name in proteins
                if paired[name.replace("protein__", "lod__", 1)].mean() < config["protein_lod_fraction"]
                and paired[name].notna().any()]
    if not retained or ko.shape[1] == 0:
        raise ValueError("KO associations require candidate KOs and proteins passing LOD QC")
    records = []
    for name in ko:
        for protein in retained:
            result = partial_spearman(ko.loc[selected.index, name], selected[protein], selected,
                                      numeric, categorical)
            records.append(dict(KO=name, protein=protein.removeprefix("protein__"), **result))
    table = pd.DataFrame(records)
    table["q"] = np.nan
    valid = table.P.notna()
    if valid.any():
        table.loc[valid, "q"] = false_discovery_control(table.loc[valid, "P"].to_numpy(), method="bh")
    table["significance"] = fdr_symbols(table.q)
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "ko_protein_partial_spearman.csv", index=False)
    metadata = dict(cohort="validation", subgroup="paired anxiety-positive MBD", n=len(selected),
                    numeric_covariates=numeric, categorical_covariates=categorical,
                    missing_values="pairwise complete observations including covariates",
                    adjustment="BH across all estimable KO-protein pairs", planned_tests=len(table),
                    estimable_tests=int(valid.sum()), significance={"*": "q < 0.10", "**": "q < 0.01"})
    (folder / "ko_protein_partial_spearman.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return table
