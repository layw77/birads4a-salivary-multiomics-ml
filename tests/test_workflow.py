import json
import shutil
import unittest
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from make_demo import generate, quick_config
from ml.data import feature_sets, read_inputs
from ml.explain import RawPredictor, explain, heldout_group_tests, probability_function
from ml.interactions import interaction_surface, pairwise_shap
from ml.associations import ko_protein_associations, partial_spearman
from ml.metrics import fdr_symbols
from ml.models import make_model, underlying
from ml.preprocessing import MicrobialCLR, ProteinQC
from ml.training import choose, train
from ml.validation import check_features, input_summary, validate_study

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "outputs" / "tests" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.config = quick_config(json.loads((ROOT / "config.json").read_text()))
        generate(self.root / "data", discovery=60, validation=60)
        self.data, self.microbes, self.proteins = read_inputs(self.root / "data", self.config)

    def test_training_only_preprocessing(self):
        training = pd.DataFrame({"a": [9, 8, 7, 6], "b": [1, 2, 3, 4], "heldout_only": [0, 0, 0, 0]})
        heldout = pd.DataFrame({"a": [0], "b": [0], "heldout_only": [1000]})
        transform = MicrobialCLR().fit(training)
        before = transform.get_feature_names_out().copy()
        result = transform.transform(heldout)
        self.assertEqual(list(before), ["a", "b"])
        np.testing.assert_array_equal(before, transform.get_feature_names_out())
        self.assertTrue(np.isfinite(result).all())
        train_protein = pd.DataFrame({"protein__a": [1, np.nan, 3, 4], "lod__a": [0, 0, 0, 0],
                                      "protein__b": [8, 9, 10, 11], "lod__b": [1, 0, 0, 0]})
        qc = ProteinQC(["protein__a", "protein__b"]).fit(train_protein)
        self.assertEqual(list(qc.get_feature_names_out()), ["protein__a"])
        test = pd.DataFrame({"protein__a": [np.nan], "lod__a": [1], "protein__b": [90], "lod__b": [0]})
        self.assertEqual(float(qc.transform(test)[0, 0]), 3.0)

    def test_asv_taxonomy_aggregation(self):
        directory = self.root / "asv"
        directory.mkdir()
        clinical = pd.read_csv(self.root / "data/clinical.csv")
        clinical.to_csv(directory / "clinical.csv", index=False)
        counts = pd.DataFrame({"patient_id": clinical.patient_id, "asv1": 3, "asv2": 4,
                               "asv3": 9, "asv4": 10, "asv5": 11})
        counts.to_csv(directory / "asv_counts.csv", index=False)
        pd.DataFrame({"asv": ["asv1", "asv2", "asv3", "asv4", "asv5"],
                      "kingdom": ["Bacteria", "Bacteria", "Eukaryota", "Bacteria", "Bacteria"],
                      "genus": ["a", "a", "b", "Chloroplast", "Mitochondria"]}).to_csv(
                          directory / "taxonomy.csv", index=False)
        frame, microbes, proteins = read_inputs(directory, self.config)
        self.assertEqual(microbes, ["micro__a"])
        self.assertEqual(proteins, [])
        self.assertTrue(frame["micro__a"].eq(7).all())

    def test_all_algorithms_and_calibrated_explanations(self):
        spec = feature_sets(self.config, self.microbes, self.proteins, True)["integrated"]
        X = self.data.loc[self.data.paired]
        y = X.outcome.to_numpy(int)
        for algorithm in self.config["algorithms"]:
            with self.subTest(algorithm=algorithm):
                params = {k: v[0] for k, v in self.config["parameters"][algorithm].items()}
                model = make_model(algorithm, params, spec, self.config, calibrated=True, top_k=7).fit(X, y)
                Z = underlying(model)[:-1].transform(X)
                self.assertEqual(Z.shape[1], 7)
                np.testing.assert_allclose(probability_function(model)(Z), model.predict_proba(X)[:, 1], atol=1e-8)
                path = self.root / f"{algorithm}.joblib"
                joblib.dump(model, path)
                np.testing.assert_allclose(joblib.load(path).predict_proba(X), model.predict_proba(X))

    def test_raw_masking_preserves_preprocessing_constraints(self):
        spec = feature_sets(self.config, self.microbes, self.proteins, True)["integrated"]
        X = self.data.loc[self.data.paired]
        model = make_model("LR", {}, spec, self.config, calibrated=True).fit(X, X.outcome)
        predict = RawPredictor(model, spec, X)
        encoded = predict.encode(X)
        np.testing.assert_allclose(predict(encoded), model.predict_proba(X)[:, 1])
        rng = np.random.default_rng(123)
        masked = encoded.iloc[:8].copy()
        donor = encoded.iloc[-8:].to_numpy()
        mask = rng.random(masked.shape) < 0.5
        masked[:] = np.where(mask, masked.to_numpy(), donor)
        raw = predict.decode(masked)
        pre = underlying(model)["preprocess"]
        clr = pre.named_transformers_["microbiome"].transform(raw[spec["microbes"]])
        np.testing.assert_allclose(clr.sum(axis=1), 0, atol=1e-12)
        category = np.asarray(pre.named_transformers_["category"].transform(raw[spec["categorical"]]))
        start = 0
        for labels in pre.named_transformers_["category"]["encode"].categories_:
            np.testing.assert_allclose(category[:, start:start + len(labels)].sum(axis=1), 1)
            start += len(labels)
        self.assertTrue(np.isfinite(predict(masked)).all())

    def test_heldout_tests_correct_entire_family(self):
        from scipy.stats import false_discovery_control
        spec = feature_sets(self.config, self.microbes, self.proteins)["clinical_bai_microbiome"]
        X = self.data.loc[self.data.cohort == "discovery"]
        training, test = X.iloc[:40], X.iloc[40:]
        model = make_model("LR", {}, spec, self.config).fit(training, training.outcome)
        tests = heldout_group_tests(model, test, test.outcome.to_numpy(), ["age"])
        self.assertEqual(len(tests), underlying(model)[:-1].transform(test).shape[1])
        self.assertTrue((tests.n_BBD + tests.n_MBD).eq(len(test)).all())
        np.testing.assert_allclose(tests.q, false_discovery_control(tests.P.to_numpy()))
        other = heldout_group_tests(model, test, test.outcome.to_numpy(), ["bai", "age", self.microbes[0]])
        np.testing.assert_allclose(tests.q, other.q)
        self.assertTrue(other.loc[other.feature == "microbiome__" + self.microbes[0], "displayed"].iloc[0])

    def test_pairwise_shap_off_diagonal_normalization(self):
        background, samples = np.zeros((1, 3)), np.array([[2., 3., 4.]])
        additive, _, _ = pairwise_shap(lambda x: x.sum(axis=1), background, samples, (0, 1), 8, 3)
        two_way, _, method = pairwise_shap(lambda x: x[:, 0] * x[:, 1], background, samples, (0, 1), 8, 3)
        three_way, _, _ = pairwise_shap(lambda x: x.prod(axis=1), background, samples, (0, 1), 8, 3)
        reverse, _, _ = pairwise_shap(lambda x: x.prod(axis=1), background, samples, (1, 0), 8, 3)
        np.testing.assert_allclose(additive, 0)
        np.testing.assert_allclose(two_way, 3)
        np.testing.assert_allclose(three_way, 6)
        np.testing.assert_allclose(reverse, three_way)
        self.assertEqual(method, "exact")
        large_sample = np.ones((1, 12))
        large_sample[:, :2] = [2, 3]
        approximate, error, method = pairwise_shap(lambda x: x[:, 0] * x[:, 1], np.zeros((1, 12)),
                                                  large_sample, (0, 1), 128, 3)
        np.testing.assert_allclose(approximate, 3)
        np.testing.assert_allclose(error, 0)
        self.assertEqual(method, "Monte Carlo")

    def test_quintic_interaction_surface_uses_unclipped_grid_statistics(self):
        rng = np.random.default_rng(3)
        measurements = rng.normal(size=(20, 2))
        interaction = measurements.prod(axis=1)
        measurements[0, 1] = np.nan
        metadata = interaction_surface(measurements, interaction, ["genus_CLR", "protein_NPX"], self.root)
        surface = pd.read_csv(self.root / "shap_interaction_surface.csv")
        self.assertTrue(np.isfinite(surface.to_numpy()).all())
        self.assertEqual(metadata["kernel"], "quintic")
        self.assertEqual(metadata["degree"], 2)
        self.assertEqual(metadata["color_limits"], [-0.06, 0.06])
        np.testing.assert_allclose(surface.SHAP_interaction, surface.genus_CLR * surface.protein_NPX, atol=1e-8)
        np.testing.assert_allclose([metadata[key] for key in ["minimum", "Q1", "median", "Q3", "maximum"]],
                                   np.quantile(surface.SHAP_interaction, [0, .25, .5, .75, 1]))
        self.assertGreater(metadata["maximum"], .06)

    def test_partial_spearman_and_fdr_symbols(self):
        from scipy.stats import spearmanr
        rng = np.random.default_rng(12)
        z = rng.normal(size=500)
        x, y = 5 * z + rng.normal(size=500), 5 * z + rng.normal(size=500)
        unadjusted = partial_spearman(x, y, pd.DataFrame(index=np.arange(len(x))), [], [])
        expected = spearmanr(x, y)
        np.testing.assert_allclose([unadjusted["rho"], unadjusted["P"]], [expected.statistic, expected.pvalue], atol=1e-12)
        adjusted = partial_spearman(x, y, pd.DataFrame({"z": z}), ["z"], [])
        self.assertLess(abs(adjusted["rho"]), .15)
        self.assertGreater(unadjusted["rho"], .9)
        self.assertEqual(adjusted["df"], 497)
        self.assertEqual(list(fdr_symbols([0.009, 0.01, 0.099, 0.10, np.nan])), ["**", "*", "*", "", ""])
        result = ko_protein_associations(self.data, self.proteins, self.root / "data", self.config,
                                         self.root / "associations")
        self.assertEqual(len(result), 6 * len(self.proteins))
        from scipy.stats import false_discovery_control
        valid = result.P.notna()
        np.testing.assert_allclose(result.loc[valid, "q"], false_discovery_control(result.loc[valid, "P"]))

    def test_preflight_detects_infeasible_nested_calibration(self):
        config = dict(self.config, outer_folds=5, inner_folds=3, calibration_folds=3,
                      algorithms=["LR", "SVM"])
        X = self.data.iloc[:30].copy()
        X["outcome"] = [0] * 25 + [1] * 5
        specs = feature_sets(config, self.microbes, self.proteins)
        with self.assertRaisesRegex(ValueError, "inner.*fit.*each class requires at least 3"):
            validate_study(X, None, specs, config, paired=True, include_explanation=False)

    def test_inputs_and_feature_schema_checks(self):
        counts, missingness = input_summary(self.data, self.proteins)
        self.assertTrue((counts.BBD + counts.MBD).eq(counts.n).all())
        self.assertTrue(missingness.missing_rate.between(0, 1).all())
        spec = feature_sets(self.config, self.microbes, self.proteins)["clinical_bai_microbiome"]
        X = self.data.loc[self.data.cohort == "discovery"]
        external = self.data.loc[self.data.cohort == "validation"]
        with self.assertRaisesRegex(ValueError, "missing features"):
            check_features(X, external.drop(columns=self.microbes[0]), spec, "validation")
        with self.assertRaisesRegex(ValueError, "schema differs"):
            check_features(X, external.assign(micro__unexpected=1), spec, "validation")
        single_class = external.copy()
        single_class["bai"] = np.where(single_class.outcome == 0, 10, 0)
        with self.assertRaisesRegex(ValueError, "anxiety-positive validation"):
            validate_study(X, single_class, {"test": spec}, self.config)
        clinical_path = self.root / "data/clinical.csv"
        clinical = pd.read_csv(clinical_path)
        for name, value in [("age", -1), ("bmi", 0), ("diameter_mm", -2), ("bai", 1.5)]:
            invalid = clinical.copy()
            invalid[name] = invalid[name].astype(float)
            invalid.loc[0, name] = value
            invalid.to_csv(clinical_path, index=False)
            with self.subTest(name=name), self.assertRaises(ValueError):
                read_inputs(self.root / "data", self.config)
        clinical.to_csv(clinical_path, index=False)
        protein_path = self.root / "data/proteomics.csv"
        npx = pd.read_csv(protein_path)
        npx.iloc[0, 1] = np.inf
        npx.to_csv(protein_path, index=False)
        with self.assertRaisesRegex(ValueError, "NPX"):
            read_inputs(self.root / "data", self.config)

    def test_selection_uses_scores(self):
        records = [dict(algorithm="LightGBM", mean_AUROC=0.7, mean_Brier=0.2, mean_log_loss=0.6, n_features=8),
                   dict(algorithm="GNB", mean_AUROC=0.85, mean_Brier=0.12, mean_log_loss=0.4, n_features=9)]
        self.assertEqual(choose(records)["algorithm"], "GNB")
        records[0]["mean_AUROC"] = 0.848
        records[0]["mean_Brier"] = 0.11
        self.assertEqual(choose(records)["algorithm"], "GNB")

    def test_temporal_and_paired_workflow(self):
        self.config["algorithms"] = ["LR", "GNB"]
        self.config["discovery_repeats"] = 2
        training = self.data.loc[self.data.cohort == "discovery"]
        validation = self.data.loc[self.data.cohort == "validation"]
        specs = feature_sets(self.config, self.microbes, self.proteins)
        result = train(training, validation, specs, self.config, self.root / "discovery")
        keys = ["patient_id", "feature_set", "algorithm", "strategy", "cohort", "subgroup"]
        self.assertFalse(result.duplicated(keys).any())
        self.assertEqual(set(result.loc[result.cohort == "validation", "patient_id"]), set(validation.index))
        search = pd.read_csv(self.root / "discovery/model_search.csv")
        self.assertEqual(int(search.loc[search.stage == "final", "selected"].sum()), 3)
        folds = pd.read_csv(self.root / "discovery/fold_predictions.csv")
        one = folds.query("cohort == 'discovery_oof' and subgroup == 'overall'")
        average = one.groupby(keys, as_index=False).probability.mean()
        check = result.query("cohort == 'discovery_oof' and subgroup == 'overall'").merge(average, on=keys)
        np.testing.assert_allclose(check.probability_x, check.probability_y)
        paired = self.data.loc[self.data.paired]
        specs = feature_sets(self.config, self.microbes, self.proteins, True)
        train(paired, None, specs, self.config, self.root / "paired", paired=True)
        deltas = pd.read_csv(self.root / "paired/paired_differences.csv")
        self.assertEqual(set(deltas.reference), {"microbiome", "proteome"})
        self.config["interaction_pair"] = ["Genus_01", "Protein_01"]
        metadata = explain(paired, specs["integrated"], self.config, self.root / "explain", calibrated=False)
        self.assertIn(metadata["algorithm"], ["LR", "GNB"])
        self.assertEqual(metadata["top_k"], 7)
        self.assertEqual(len(metadata["selected_features"]), 7)
        self.assertEqual(metadata["interaction"]["status"], "computed")
        interactions = pd.read_csv(self.root / "explain/pairwise_shap_interactions.csv")
        self.assertEqual(len(interactions), metadata["explanation_n"])
        self.assertTrue(np.isfinite(interactions.SHAP_interaction).all())
        with np.load(self.root / "explain/shap.npz") as shap:
            np.testing.assert_allclose(shap["base"] + shap["values"].sum(axis=1), shap["probability"], atol=1e-6)
        group_tests = pd.read_csv(self.root / "explain/feature_group_tests.csv")
        self.assertTrue((group_tests.n_BBD + group_tests.n_MBD).eq(metadata["explanation_n"]).all())
        self.assertEqual(set(pd.read_csv(self.root / "paired/predictions.csv").strategy), {"full", "top7"})
        schema = json.loads((self.root / "explain/raw_schema.json").read_text())
        self.assertEqual(set(schema["categorical_codes"]), set(self.config["clinical_categorical"]))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main()
