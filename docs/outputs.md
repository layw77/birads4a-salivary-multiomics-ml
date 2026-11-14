# Outputs

Results are written beneath `outputs/<run>/discovery/` and `outputs/<run>/paired/`.

| File | Contents |
|---|---|
| `folds.csv` | Outer fold membership |
| `model_search.csv` | Inner-fold scores, candidate parameters and selected candidates |
| `fold_predictions.csv` | Predictions for each outer repeat/fold and temporal-validation predictions |
| `predictions.csv` | Participant-level probabilities; repeated OOF estimates are averaged |
| `metrics.csv` | Classification metrics and bootstrap AUROC/AP intervals |
| `roc_bands.csv` | Bootstrap pointwise ROC intervals |
| `paired_differences.csv` | Paired differences in AUROC and AP between predictor sets |
| `selected_features.csv` | Features of each outer-fold selected model |
| `fold_importance.csv` | Training-fold importance of the selected estimator |
| `importance.csv` | Mean normalized importance across outer folds |
| `selected_models.json` | Final selected algorithms, parameters and feature names |
| `preflight.json` | Outcome counts and feasibility of planned fitting partitions |
| `models/` | Fitted preprocessing, selection, classifier and calibration objects |
| `figures/` | ROC/PR curves, confusion matrices, probability distributions and incremental performance |

Candidate estimator importance is absolute coefficient magnitude for linear models, native importance for tree models, and permutation AUROC importance for other estimators. The same definition supplies fold-local Top-K ranking. Importance is normalized within each selected outer-fold model before averaging; features absent from a fold receive zero weight.

Bootstrap intervals resample participant predictions 2,000 times. The same sampled participants are used for each predictor set in paired differences. Discovery and validation intervals are calculated separately. These intervals summarize prediction resampling; model fitting occurs in the cross-validation procedure.

The `selected` row evaluates the complete algorithm-selection procedure, including different winners in different outer folds. The final selected model is refitted after inner cross-validation on the full development sample. Each predictor set has its own selection. `top7` denotes selection of at most seven transformed predictors, including any selected clinical or BAI terms.

`explanations/<feature_set>/` contains the 80:20 model, split membership, training search results, SHAP arrays, importance, dependence and waterfall figures. Algorithm and parameter selection use only the 80% training sample; paired models also select Top7 within training folds. Background values come from that sample. SHAP values sum with the baseline to the selected model's MBD probability. `model.json` records the algorithm and selected transformed predictors. Recomputing CLR can attribute contributions to additional raw genera through its shared denominator.

SHAP masking operates on original numeric variables, individual categorical variables, genus counts and NPX measurements. Categorical integer codes are transport labels for complete category values; `raw_schema.json` records their mapping and the raw feature order. Every masked sample passes through the fitted preprocessing, so CLR values are recalculated jointly and one-hot categories remain mutually exclusive. LOD flags determine training QC and are not SHAP predictors. These are marginal attributions; correlations among original variables can make some replacement combinations uncommon.

`feature_group_tests.csv` contains Mann–Whitney comparisons for every transformed predictor before Top7 selection, using only the 20% holdout. BH adjustment is calculated over this entire family. `displayed_feature_group_tests.csv` selects display rows after correction. P/q values quantify pathology-group differences in held-out features. SHAP values quantify prediction attributions.

For the integrated classifier, `pairwise_shap_interactions.csv` records the Fusobacterium–CCL3 off-diagonal interaction for each holdout participant. The requested matrix element is computed directly as one-half of the Shapley-weighted second coalition difference, integrating over the training background. All coalitions are enumerated when they fit within `interaction_draws`; otherwise uniformly sampled coalition sizes estimate the same element. Sampling errors and the method are recorded. The selected classifier and its complete fitted preprocessing supply MBD probabilities for every coalition.

`shap_interaction_surface.csv` contains the fitted 60 × 60 grid over Fusobacterium CLR abundance and CCL3 NPX. A regularized quintic RBF uses a degree-two polynomial and standardized coordinates. Filled contours and iso-interaction lines use a zero-centered color scale saturated at ±0.060. `shap_interaction_surface.json` reports quartiles, median and extrema of the full fitted grid. The configured pair is analyzed when its input features pass training QC.

`associations/ko_protein_partial_spearman.csv` reports covariate-adjusted KO–protein correlations in paired anxiety-positive MBD participants. Ranked measurements and continuous covariates are residualized against the covariate design; categorical covariates use indicator variables. Two-sided P values use residual degrees of freedom. BH adjustment covers all estimable KO–protein pairs in one family. The accompanying JSON records covariates and sample counts. All significance columns use `*` for FDR-adjusted P <0.10 and `**` for FDR-adjusted P <0.01.

At run level, `cohort_counts.csv` summarizes BBD/MBD counts overall, among anxiety-positive participants and among paired anxiety-positive participants. `protein_missingness.csv` reports missingness and below-LOD rates. Stage run JSON files record effective configuration, Python and package versions, and SHA-256 input checksums. These files remain in the local output directory.
