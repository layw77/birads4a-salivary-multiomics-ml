import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import false_discovery_control, mannwhitneyu
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .models import underlying
from .interactions import interaction_surface, pairwise_shap
from .metrics import fdr_symbols
from .training import fit_record, record_search, tune
from .validation import validate_study


def display_name(name):
    for prefix in ["clinical__", "category__", "microbiome__micro__", "proteome__protein__", "micro__", "protein__"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return "BAI score" if name == "bai" else name


def probability_function(model):
    base = underlying(model)
    clf = base["classifier"]
    names = base["select"].get_feature_names_out()

    def predict(values):
        frame = pd.DataFrame(values, columns=names)
        if isinstance(model, CalibratedClassifierCV):
            response = clf.decision_function(frame) if hasattr(clf, "decision_function") else clf.predict_proba(frame)[:, 1]
            return model.calibrated_classifiers_[0].calibrators[0].predict(response)
        return clf.predict_proba(frame)[:, 1]

    return predict


class RawPredictor:
    def __init__(self, model, spec, reference):
        self.model = model
        self.spec = spec
        self.names = spec["numeric"] + spec["categorical"] + spec["microbes"] + spec["proteins"]
        self.categories = {name: reference[name].dropna().unique().tolist() for name in spec["categorical"]}

    def encode(self, frame):
        encoded = frame[self.names].copy()
        for name, labels in self.categories.items():
            encoded[name] = frame[name].map({label: i for i, label in enumerate(labels)}).astype(float)
        return encoded.astype(float)

    def decode(self, values):
        frame = pd.DataFrame(np.asarray(values), columns=self.names)
        for name, labels in self.categories.items():
            frame[name] = frame[name].map(dict(enumerate(labels))).astype(object)
        for name in self.spec["proteins"]:
            frame[name.replace("protein__", "lod__", 1)] = 0
        return frame

    def __call__(self, values):
        return self.model.predict_proba(self.decode(values))[:, 1]


def heldout_group_tests(model, heldout, outcome, shown):
    transformed = underlying(model)[:2].transform(heldout)
    comparisons = []
    for name in transformed:
        benign = transformed.loc[np.asarray(outcome) == 0, name]
        malignant = transformed.loc[np.asarray(outcome) == 1, name]
        statistic, p = mannwhitneyu(benign, malignant, alternative="two-sided")
        comparisons.append(dict(feature=name, U=statistic, P=p, n_BBD=len(benign), n_MBD=len(malignant)))
    comparison = pd.DataFrame(comparisons)
    comparison["q"] = false_discovery_control(comparison.P.to_numpy(), method="bh")
    comparison["significance"] = fdr_symbols(comparison.q)
    shown_names = {display_name(name) for name in shown}
    comparison["displayed"] = comparison.feature.map(lambda name:
        display_name(name) in shown_names or
        any(name.startswith("category__" + feature + "_") for feature in shown_names))
    comparison["sample"] = "explanation_holdout"
    return comparison


def explain(X, spec, config, folder, calibrated):
    validate_study(X, None, {"explanation": spec}, config, paired=not calibrated,
                   include_training=False, include_explanation=True)
    folder.mkdir(parents=True, exist_ok=True)
    y = X.outcome.to_numpy(int)
    tr, te = train_test_split(np.arange(len(X)), test_size=config["shap_test_fraction"],
                             stratify=y, random_state=config["seed"])
    top_k = None if calibrated else config["top_k"][0]
    chosen, _, records = tune(X.iloc[tr], y[tr], spec, config, calibrated, top_k)
    model = fit_record(chosen, X.iloc[tr], y[tr], spec, config, calibrated, top_k)
    predict = RawPredictor(model, spec, X)
    training = predict.encode(X.iloc[tr])
    test = predict.encode(X.iloc[te])
    np.testing.assert_allclose(predict(test), model.predict_proba(X.iloc[te])[:, 1], atol=1e-8)
    background = training.sample(min(config["shap_background"], len(training)), random_state=config["seed"])
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(predict, masker, algorithm="permutation", seed=config["seed"])
    evaluations = (2 * test.shape[1] + 1) * config["shap_permutations"]
    values = explainer(test, max_evals=evaluations)
    np.testing.assert_allclose(values.base_values + values.values.sum(axis=1), predict(test), atol=1e-6)
    names = np.asarray(test.columns, dtype=str)
    magnitude = np.abs(values.values).mean(axis=0)
    importance = pd.DataFrame({"feature": names, "mean_abs_SHAP": magnitude,
                               "percent": 100 * magnitude / magnitude.sum() if magnitude.sum() else 0})
    importance = importance.sort_values("mean_abs_SHAP", ascending=False)
    importance.to_csv(folder / "shap_importance.csv", index=False)
    np.savez_compressed(folder / "shap.npz", values=values.values, base=values.base_values,
                        data=test.to_numpy(), features=names, patient_id=X.index.to_numpy(str)[te],
                        outcome=y[te], probability=predict(test))
    split = pd.DataFrame({"patient_id": X.index, "role": "training"})
    split.loc[te, "role"] = "explanation"
    split.to_csv(folder / "split.csv", index=False)
    pd.DataFrame(record_search(records, chosen, {})).to_csv(folder / "model_search.csv", index=False)
    metadata = dict(algorithm=chosen["algorithm"], params=chosen["params"],
                    top_k=top_k, selected_features=underlying(model)["select"].get_feature_names_out().tolist(),
                    training_n=len(tr), explanation_n=len(te), output="MBD probability",
                    masking="Raw predictors; full fitted preprocessing on each masked sample",
                    attribution="Marginal SHAP with training-background replacement",
                    group_tests="Held-out participants; BH across all training-defined transformed predictors",
                    holdout_AUROC=roc_auc_score(y[te], predict(test)))
    (folder / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    schema = dict(features=predict.names, categorical_codes=predict.categories)
    (folder / "raw_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    joblib.dump(model, folder / "model.joblib", compress=3)

    display = shap.Explanation(values=values.values, base_values=values.base_values,
                               data=values.data, feature_names=[display_name(name) for name in names])
    shap.plots.beeswarm(display, max_display=config["display_features"], show=False)
    plt.title(f"{chosen['algorithm']} | MBD probability")
    save_plot(folder / "shap_summary")
    shap.plots.bar(display, max_display=config["display_features"], show=False)
    save_plot(folder / "shap_importance")
    shap.plots.waterfall(display[0], max_display=config["display_features"], show=False)
    save_plot(folder / "shap_waterfall")
    shown = importance.head(config["display_features"]).feature.tolist()
    for rank, name in enumerate(shown, 1):
        shap.plots.scatter(display[:, list(names).index(name)], color=display, show=False)
        save_plot(folder / f"dependence_{rank:02d}")
    comparison = heldout_group_tests(model, X.iloc[te], y[te], shown)
    comparison.to_csv(folder / "feature_group_tests.csv", index=False)
    comparison.loc[comparison.displayed].to_csv(folder / "displayed_feature_group_tests.csv", index=False)
    if spec["microbes"] and spec["proteins"]:
        metadata["interaction"] = explain_interaction(predict, background, test, X.iloc[te], config, folder)
        (folder / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def save_plot(path):
    fig = plt.gcf()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def explain_interaction(predict, background, test, heldout, config, folder):
    pair = config["interaction_pair"]
    microbe, protein = "micro__" + pair[0], "protein__" + pair[1]
    if microbe not in predict.names or protein not in predict.names:
        return dict(status="pair_unavailable", pair=pair)
    base = underlying(predict.model)
    transform = base["preprocess"].named_transformers_["microbiome"]
    microbial_names = list(transform.get_feature_names_out())
    if microbe not in microbial_names:
        return dict(status="microbe_excluded_by_training_QC", pair=pair)
    indices = (predict.names.index(microbe), predict.names.index(protein))
    interaction, error, method = pairwise_shap(predict, background, test, indices,
                                              config["interaction_draws"], config["seed"])
    clr = np.asarray(transform.transform(heldout[predict.spec["microbes"]]))[:, microbial_names.index(microbe)]
    coordinates = np.column_stack([clr, heldout[protein].to_numpy(float)])
    pd.DataFrame({"patient_id": heldout.index, pair[0] + "_CLR": clr,
                  pair[1] + "_NPX": coordinates[:, 1], "SHAP_interaction": interaction,
                  "Monte_Carlo_SE": error, "feature_i": microbe, "feature_j": protein}).to_csv(
        folder / "pairwise_shap_interactions.csv", index=False)
    metadata = dict(status="computed", pair=pair, method=method, draws=config["interaction_draws"],
                    background_n=len(background), output="MBD probability",
                    element="single off-diagonal; one-half weighted second coalition difference",
                    masking="Raw predictors with full fitted preprocessing")
    metadata.update(interaction_surface(coordinates, interaction, [pair[0] + " (CLR)", pair[1] + " (NPX)"],
                                        folder, config["interaction_rbf_smoothing"]))
    return metadata
