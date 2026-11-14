from pathlib import Path
import csv

import numpy as np
import pandas as pd


def read_matrix(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)) or "patient_id" not in header:
        raise ValueError(f"{Path(path).name}: unique column names and patient_id are required")
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        raise ValueError(f"{Path(path).name}: patient_id must be unique and complete")
    return frame.set_index("patient_id")


def read_inputs(directory, config):
    directory = Path(directory)
    clinical = read_matrix(directory / "clinical.csv")
    required = ["outcome", "cohort", "bai", *config["clinical_numeric"],
                *config["clinical_categorical"]]
    missing = set(required) - set(clinical)
    if missing:
        raise ValueError(f"clinical.csv: missing columns {sorted(missing)}")
    if not clinical.outcome.isin([0, 1]).all():
        raise ValueError("outcome must be 0 (BBD) or 1 (MBD)")
    if not clinical.cohort.isin(["discovery", "validation"]).all():
        raise ValueError("cohort must be discovery or validation")
    if not clinical.bai.between(0, 63).all():
        raise ValueError("bai must be between 0 and 63")
    if not clinical.bai.eq(clinical.bai.round()).all():
        raise ValueError("bai must contain integer scores")
    for name in config["clinical_numeric"]:
        clinical[name] = pd.to_numeric(clinical[name], errors="raise")
        observed = clinical[name].dropna()
        if not np.isfinite(observed).all():
            raise ValueError(f"{name}: measurements must be finite")
        invalid = ((observed <= 0) | (observed > 120)) if name == "age" else (
            observed <= 0 if name == "bmi" else observed < 0 if name == "diameter_mm" else pd.Series(False, index=observed.index))
        if invalid.any():
            raise ValueError(f"{name}: invalid clinical value")
    clinical[config["clinical_categorical"]] = clinical[config["clinical_categorical"]].astype(object)

    if (directory / "genus_counts.csv").exists():
        counts = read_matrix(directory / "genus_counts.csv")
    else:
        counts = read_matrix(directory / "asv_counts.csv")
        taxonomy = pd.read_csv(directory / "taxonomy.csv").set_index("asv")
        taxonomy = taxonomy.loc[counts.columns]
        bacterial = taxonomy.kingdom.str.replace(r"^[dk]__", "", regex=True).str.lower().eq("bacteria")
        organelle = taxonomy.astype(str).apply(
            lambda col: col.str.contains("chloroplast|mitochondri", case=False, regex=True)
        ).any(axis=1)
        taxonomy = taxonomy.loc[bacterial & ~organelle & taxonomy.genus.notna()]
        counts = counts[taxonomy.index].T.groupby(taxonomy.genus, sort=False).sum().T
    values = counts.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values.sum(axis=1) <= 0).any():
        raise ValueError("Microbial counts must be finite, nonnegative and have positive row totals")
    if not clinical.index.isin(counts.index).all():
        raise ValueError("Microbial counts are required for each clinical participant")
    data = clinical.join(counts.add_prefix("micro__"), how="left", validate="one_to_one")
    proteins = []
    if (directory / "proteomics.csv").exists():
        npx = read_matrix(directory / "proteomics.csv")
        lod = read_matrix(directory / "protein_lod.csv")
        npx = npx.apply(pd.to_numeric, errors="raise")
        if np.isinf(npx.to_numpy(float)).any():
            raise ValueError("Protein NPX measurements must be finite or missing")
        if set(npx.columns) != set(lod.columns) or set(npx.index) != set(lod.index):
            raise ValueError("Protein NPX and LOD tables must have matching participants and features")
        lod = lod.loc[npx.index, npx.columns]
        if not lod.isin([0, 1]).all().all():
            raise ValueError("Protein LOD flags must be 0 or 1")
        if not npx.index.isin(clinical.index).all():
            raise ValueError("Protein participants must be present in clinical.csv")
        data = data.join(npx.add_prefix("protein__"), validate="one_to_one")
        data = data.join(lod.add_prefix("lod__"), validate="one_to_one")
        data["paired"] = data.index.isin(npx.index)
        proteins = ["protein__" + name for name in npx]
    else:
        data["paired"] = False
    return data, ["micro__" + name for name in counts], proteins


def feature_sets(config, microbes, proteins, paired=False):
    numeric = config["clinical_numeric"]
    categorical = config["clinical_categorical"]
    base = {"numeric": numeric, "categorical": categorical, "microbes": [], "proteins": []}
    with_bai = dict(base, numeric=numeric + ["bai"])
    if paired:
        return {
            "microbiome": dict(with_bai, microbes=microbes),
            "proteome": dict(with_bai, proteins=proteins),
            "integrated": dict(with_bai, microbes=microbes, proteins=proteins),
        }
    return {"clinical": base, "clinical_bai": with_bai,
            "clinical_bai_microbiome": dict(with_bai, microbes=microbes)}
