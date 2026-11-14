import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance


class MicrobialCLR(TransformerMixin, BaseEstimator):
    def __init__(self, prevalence=0.1, abundance=0.0001, pseudocount=0.5):
        self.prevalence = prevalence
        self.abundance = abundance
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        a = np.asarray(X, dtype=float)
        relative = a / a.sum(axis=1, keepdims=True)
        self.retained_ = ((a > 0).mean(axis=0) >= self.prevalence) & (relative.mean(axis=0) >= self.abundance)
        z = self._clr(a)
        self.variable_ = np.var(z, axis=0) > 0
        return self

    def _clr(self, a):
        z = np.log(a[:, self.retained_] + self.pseudocount)
        if z.shape[1]:
            z -= z.mean(axis=1, keepdims=True)
        return z

    def transform(self, X):
        if hasattr(X, "columns"):
            missing = set(self.feature_names_in_) - set(X.columns)
            if missing:
                raise ValueError(f"Missing microbial features: {sorted(missing)}")
            X = X.loc[:, self.feature_names_in_]
        values = np.asarray(X, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Microbial features must be finite and nonnegative")
        return self._clr(values)[:, self.variable_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.retained_][self.variable_]


class ProteinQC(TransformerMixin, BaseEstimator):
    def __init__(self, proteins, lod_fraction=0.25):
        self.proteins = proteins
        self.lod_fraction = lod_fraction

    def fit(self, X, y=None):
        names = np.asarray(self.proteins, dtype=object)
        lod = [name.replace("protein__", "lod__", 1) for name in names]
        self.missing_rate_ = X[names].isna().mean().to_numpy()
        self.names_ = names[(X[lod].to_numpy(float).mean(axis=0) < self.lod_fraction) &
                            (self.missing_rate_ < 1)]
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        if len(self.names_):
            self.imputer_.fit(X[self.names_])
        return self

    def transform(self, X):
        if not len(self.names_):
            return np.empty((len(X), 0))
        return self.imputer_.transform(X[self.names_])

    def get_feature_names_out(self, input_features=None):
        return self.names_


class RankedTopK(TransformerMixin, BaseEstimator):
    def __init__(self, estimator, k=None, seed=2026):
        self.estimator = estimator
        self.k = k
        self.seed = seed

    def fit(self, X, y):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if self.k is None:
            self.selected_ = np.arange(X.shape[1])
        else:
            if not isinstance(self.k, int) or self.k < 1:
                raise ValueError("Top-K must be a positive integer")
            fitted = clone(self.estimator).fit(X, y)
            importance = model_importance(fitted, X, y, self.seed)
            self.selected_ = np.argsort(-importance, kind="stable")[:self.k]
        return self

    def transform(self, X):
        return X.iloc[:, self.selected_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.selected_]


def model_importance(estimator, X, y, seed):
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_)
    if hasattr(estimator, "coef_"):
        return np.abs(estimator.coef_).mean(axis=0)
    return permutation_importance(estimator, X, y, scoring="roc_auc", n_repeats=5,
                                  random_state=seed, n_jobs=1).importances_mean
