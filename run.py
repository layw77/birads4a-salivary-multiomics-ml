import argparse
import importlib.metadata
import hashlib
import json
import platform
from pathlib import Path

from threadpoolctl import threadpool_limits

from ml.data import feature_sets, read_inputs
from ml.associations import ko_protein_associations
from ml.explain import explain
from ml.plots import plot_results
from ml.training import train
from ml.validation import input_summary, validate_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validate", "train", "explain", "figures", "associations", "all"])
    parser.add_argument("--data", type=Path, default=Path("data/local"))
    parser.add_argument("--output", type=Path, default=Path("outputs/local"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--study", choices=["discovery", "paired", "both"], default="both")
    args = parser.parse_args()
    saved_config = args.output / "config.json"
    config_path = args.config or (saved_config if args.stage in ["explain", "figures", "associations"] and saved_config.exists()
                                 else Path(__file__).with_name("config.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in [
        "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "shap", "matplotlib"]}
    versions.update(python=platform.python_version(), numba=importlib.metadata.version("numba"),
                    llvmlite=importlib.metadata.version("llvmlite"))
    if args.stage in ["validate", "train", "all"]:
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (args.output / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    studies = ["discovery", "paired"] if args.study == "both" else [args.study]
    if args.stage != "figures":
        data, microbes, proteins = read_inputs(args.data, config)
        counts, missingness = input_summary(data, proteins)
        counts.to_csv(args.output / "cohort_counts.csv", index=False)
        missingness.to_csv(args.output / "protein_missingness.csv", index=False)
        print(counts.to_string(index=False), flush=True)
        manifest = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                    for file in sorted(args.data.glob("*.csv"))}
        metadata = dict(stage=args.stage, config=config, versions=versions, input_sha256=manifest)
        (args.output / f"{args.stage}_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.stage == "associations":
            ko_protein_associations(data, proteins, args.data, config, args.output / "associations")
            return
    tasks = []
    for study in studies:
        paired = study == "paired"
        folder = args.output / study
        if args.stage != "figures":
            if paired:
                X = data.loc[(data.cohort == "validation") & (data.bai >= 8) & data.paired].copy()
                external = None
                if not proteins or X.empty:
                    raise ValueError("The paired study requires anxiety-positive validation participants with proteomics")
            else:
                X = data.loc[data.cohort == "discovery"].copy()
                external = data.loc[data.cohort == "validation"].copy()
                if X.empty or external.empty:
                    raise ValueError("The discovery study requires discovery and validation cohorts")
            specs = feature_sets(config, microbes, proteins, paired)
            report = validate_study(X, external, specs, config, paired,
                                    include_training=args.stage != "explain",
                                    include_explanation=args.stage in ["validate", "explain", "all"])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            tasks.append((paired, folder, X, external, specs))
        else:
            tasks.append((paired, folder, None, None, None))
    for paired, folder, X, external, specs in tasks:
        if args.stage in ["train", "all"]:
            train(X, external, specs, config, folder, paired)
        if args.stage in ["explain", "all"]:
            requested = specs if paired else {"clinical_bai_microbiome": specs["clinical_bai_microbiome"]}
            for name, spec in requested.items():
                explain(X, spec, config, folder / "explanations" / name, calibrated=not paired)
        if args.stage in ["figures", "all"]:
            plot_results(folder)
    if args.stage == "all" and "paired" in studies and (args.data / "ko_abundances.csv").exists():
        ko_protein_associations(data, proteins, args.data, config, args.output / "associations")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
