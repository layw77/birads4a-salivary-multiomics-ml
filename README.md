# Salivary multiomics Signatures for Malignancy Prediction in BI-RADS 4A Breast Lesions

## Install

Python 3.14 (64-bit), tested with 3.14.2.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```bash
python make_demo.py --output data/demo
python run.py all --data data/demo --output outputs/demo
```

For a shorter example with fewer folds, parameter combinations and bootstrap resamples:

```bash
python make_demo.py --output data/quick --quick
python run.py all --data data/quick --config data/quick/quick_config.json --output outputs/quick
```

Study inputs are supplied locally in the format described in [docs/inputs.md](docs/inputs.md).

```bash
python run.py train --data data/local --output outputs/local
python run.py explain --data data/local --output outputs/local
python run.py figures --output outputs/local
python run.py associations --data data/local --output outputs/local
```

Use `--study discovery` or `--study paired` to run one analysis. Analysis settings and parameter grids are in [config.json](config.json).

Check inputs and all planned cross-validation partitions before fitting:

```bash
python run.py validate --data data/local --output outputs/local
```

## Analyses

- **Discovery and temporal validation:** Clinical, Clinical + BAI and Clinical + BAI + Microbiome models. Five outer folds repeated three times, three inner folds, training-only sigmoid calibration, and final fitting on the discovery cohort. Results include the overall population and the BAI ≥8 subgroup.
- **Paired multiomics:** Microbiome, proteome and integrated models in anxiety-positive validation participants with paired assays. Each model contains the same clinical and BAI variables. Five outer folds and three inner folds compare full and Top7 feature sets.
- **Algorithms:** LR, SVM, Gaussian naïve Bayes, MLP, random forest, XGBoost and LightGBM. The highest mean inner-fold AUROC selects the algorithm and hyperparameters.
- **Preprocessing:** Training-fold genus prevalence and mean relative-abundance filtering, CLR transformation, median imputation, categorical mode imputation and one-hot encoding, and robust scaling. Protein LOD filtering is fitted on each training partition; measured below-LOD NPX values are retained. Top-K rankings use the candidate estimator's training-fold importance.
- **Evaluation:** AUROC, AP, accuracy, sensitivity, specificity, PPV, NPV and F1 at a probability threshold of 0.5. Repeated OOF predictions are averaged by participant. Paired participant bootstrap resampling supplies AUROC/AP intervals, ROC bands and differences between feature sets.
- **SHAP:** A stratified 80:20 split selects and fits the model on the training partition, using Top7 for paired multiomics. Raw-predictor masking passes through the fitted pipeline, recalculating CLR and categorical encoding. Display order follows computed importance.
- **Interactions:** The integrated model supplies the Fusobacterium–CCL3 off-diagonal SHAP interaction. A regularized quintic RBF with a quadratic polynomial fits the CLR–NPX surface; the color scale spans ±0.060. Grid statistics accompany the figure.
- **Statistics:** Held-out Mann–Whitney comparisons use BH adjustment across all training-defined transformed predictors. KO–protein partial Spearman correlations use paired anxiety-positive MBD participants, adjust for clinical covariates and BAI, and form a separate BH family. Significance: `*` for FDR-adjusted P <0.10; `**` for FDR-adjusted P <0.01.

`selected` in performance tables denotes the nested algorithm-selection procedure. Algorithm-named rows evaluate each algorithm with its own inner-fold parameter selection. Final algorithm names and features are recorded in `selected_models.json`; the 80:20 explanation model is recorded separately in `explanations/*/model.json`.

## Files

| Path | Contents |
|---|---|
| `run.py` | Training, explanation and figure commands |
| `ml/` | Input handling, preprocessing, models, cross-validation and plotting |
| `make_demo.py` | Synthetic input generator |
| `config.json` | Analysis settings and seven algorithm grids |
| `docs/` | Input and output definitions |
| `tests/` | Synthetic workflow checks |

Participant data, fitted models and analysis outputs are stored locally in the ignored `data/` and `outputs/` directories. The repository contains code and a synthetic data generator.

Preflight outputs include cohort outcome counts, protein missingness and LOD rates, and class counts throughout outer, inner and calibration partitions. Run metadata records input checksums, settings and package versions. Separate explanation and figure commands use the saved run configuration unless `--config` is supplied.

```bash
python -m unittest discover -s tests -v
```
