from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .preprocessing import MicrobialCLR, ProteinQC, RankedTopK


def estimator(name, params, seed):
    constructors = {
        "LR": (LogisticRegression, dict(max_iter=5000, random_state=seed)),
        "SVM": (SVC, dict(random_state=seed)),
        "GNB": (GaussianNB, {}),
        "MLP": (MLPClassifier, dict(hidden_layer_sizes=(32,), alpha=1.0, max_iter=5000, random_state=seed)),
        "RF": (RandomForestClassifier, dict(n_estimators=500, n_jobs=1, random_state=seed)),
        "XGBoost": (XGBClassifier, dict(max_depth=3, learning_rate=0.1, n_estimators=200,
                                      eval_metric="logloss", n_jobs=1, random_state=seed)),
        "LightGBM": (LGBMClassifier, dict(max_depth=4, num_leaves=15, learning_rate=0.05,
                                        n_estimators=300, verbosity=-1, n_jobs=1, random_state=seed)),
    }
    constructor, defaults = constructors[name]
    arguments = defaults | params
    if "hidden_layer_sizes" in arguments:
        arguments["hidden_layer_sizes"] = tuple(arguments["hidden_layer_sizes"])
    return constructor(**arguments)


def make_model(name, params, spec, config, calibrated=False, top_k=None):
    seed = config["seed"]
    branches = [
        ("clinical", SimpleImputer(strategy="median", keep_empty_features=True), spec["numeric"]),
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), spec["categorical"]),
    ]
    if spec["microbes"]:
        branches.append(("microbiome", MicrobialCLR(config["microbial_prevalence"],
                         config["microbial_mean_abundance"], config["count_pseudocount"]), spec["microbes"]))
    if spec["proteins"]:
        lod = [name.replace("protein__", "lod__", 1) for name in spec["proteins"]]
        branches.append(("proteome", ProteinQC(spec["proteins"], config["protein_lod_fraction"]),
                         spec["proteins"] + lod))
    preprocessor = ColumnTransformer(branches, sparse_threshold=0).set_output(transform="pandas")
    clf = estimator(name, params, seed)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("scale", RobustScaler().set_output(transform="pandas")),
        ("select", RankedTopK(clf, top_k, seed)),
        ("classifier", clf),
    ])
    if calibrated or name == "SVM":
        folds = StratifiedKFold(config["calibration_folds"], shuffle=True, random_state=seed + 19)
        return CalibratedClassifierCV(pipeline, method="sigmoid", cv=folds, ensemble=False, n_jobs=1)
    return pipeline


def underlying(model):
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model
