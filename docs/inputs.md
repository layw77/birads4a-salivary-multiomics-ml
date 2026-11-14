# Input tables

UTF-8 CSV files use one row per participant and an anonymous `patient_id`. Clinical participants have BI-RADS 4A lesions and a histopathological outcome. Healthy controls are omitted from classification. IDs link the clinical, microbial and protein tables.

| File | Columns |
|---|---|
| `clinical.csv` | `patient_id`, `cohort`, `outcome`, `bai`, clinical variables configured in `config.json` |
| `genus_counts.csv` | `patient_id`, one column per bacterial genus containing nonnegative read counts |
| `proteomics.csv` | `patient_id`, one column per assayed protein containing observed log2 NPX values |
| `protein_lod.csv` | Same participant IDs and protein columns as `proteomics.csv`; 1 below LOD, 0 otherwise |
| `ko_abundances.csv` | `patient_id`, one column per predicted KO; required for `associations` |

`cohort` is `discovery` or `validation`. `outcome` is 0 for BBD and 1 for MBD. `bai` is the BAI total score (0–63); anxiety positivity is BAI ≥8. Numeric defaults are `age`, `bmi` and `diameter_mm`. Categorical defaults are `menopause`, `family_history`, `smoking`, `alcohol`, `shape`, `orientation` and `margin`; category labels are encoded during training. Empty clinical or NPX fields represent missing measurements.

Protein tables contain the validation participants with paired salivary assays. Supply all measured candidate proteins and their LOD flags. The workflow fits the ≥25% below-LOD exclusion on training data and imputes missing retained NPX values during model fitting. Proteomics files are needed for `--study paired` and `--study both`.

Column names and participant IDs must be unique. Numeric measurements must be finite, apart from permitted missing clinical and NPX values. Age is greater than 0 and at most 120 years, BMI is positive, lesion diameter is nonnegative, and BAI is an integer from 0 to 63. Training and evaluation use the same assay feature schema; categorical levels first encountered in evaluation are handled by the fitted encoder.

Protein missingness is reported per feature. Entirely missing training proteins are omitted because their medians cannot be estimated. No additional missing-rate cutoff is applied. Each training partition must retain measured proteins after LOD QC, and clinical imputation variables must have observed training values. Class-count checks follow every actual outer, inner, calibration and explanation split, using the configured folds and seeds. A partition that cannot support the requested folds raises an error before model training.

KO abundances are nonnegative predicted abundances from PICRUSt2 and include every paired anxiety-positive MBD participant. Missing measurements use pairwise complete observations. Association covariates are age, BMI, BAI, menopausal status, smoking, alcohol consumption and family history; names are configured in `association_numeric` and `association_categorical`. Protein LOD QC uses all paired anxiety-positive validation participants before restricting correlations to MBD. `all` runs these correlations when the KO table is present.

As an alternative to `genus_counts.csv`, supply:

| File | Columns |
|---|---|
| `asv_counts.csv` | `patient_id`, one count column per ASV |
| `taxonomy.csv` | `asv`, `kingdom`, `genus`; additional taxonomy columns may be included |

ASVs assigned outside Bacteria, chloroplasts or mitochondria are removed before aggregation to genus counts. `genus_counts.csv` takes precedence when both formats are present.

Genus filtering uses training prevalence ≥10% and mean relative abundance ≥0.01%. CLR values are calculated as `log(count + 0.5)` minus the row mean over retained genera. Invariant training CLR features are removed. The trained transformation is applied to the corresponding evaluation partition.

Predictor groups are defined by clinical variables and assay membership. Molecular feature identities are read from the input columns; no named taxon or protein panel is embedded in the model code.

`interaction_pair` specifies the post-fit Fusobacterium–CCL3 analysis. It does not enter model or feature selection. `interaction_draws` controls coalition sampling and `interaction_rbf_smoothing` controls RBF regularization.
