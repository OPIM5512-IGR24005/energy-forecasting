# tomato_range.py: Tomato's final project range model (OPIM 5512).
# ONE scikit-learn model whose predict(X) returns 3 columns: low (10th percentile), expected, high (90th percentile).
# Inside it are three gradient boosting models trained on the same inputs:
#   low  = gradient boosting with a quantile loss at 0.1
#   mid  = gradient boosting with the usual squared-error loss (the expected value)
#   high = gradient boosting with a quantile loss at 0.9
# This file must sit next to main.py in cloud_function/forecast-24h, so the cloud can load the saved model.
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor


class ThreeQuantileBoost(RegressorMixin, BaseEstimator):
    def __init__(self, learning_rate=0.02, max_iter=2000, max_leaf_nodes=15, min_samples_leaf=50,
                 l2_regularization=1.0, max_features=0.85, max_depth=8, low_q=0.1, high_q=0.9, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.max_depth = max_depth
        self.low_q = low_q
        self.high_q = high_q
        self.random_state = random_state

    def _one(self, **loss):
        return HistGradientBoostingRegressor(learning_rate=self.learning_rate, max_iter=self.max_iter,
                                             max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf,
                                             l2_regularization=self.l2_regularization, max_features=self.max_features,
                                             max_depth=self.max_depth, early_stopping=False,
                                             random_state=self.random_state, **loss)

    def fit(self, X, y, sample_weight=None):
        self.low_ = self._one(loss="quantile", quantile=self.low_q).fit(X, y, sample_weight=sample_weight)
        self.mid_ = self._one().fit(X, y, sample_weight=sample_weight)
        self.high_ = self._one(loss="quantile", quantile=self.high_q).fit(X, y, sample_weight=sample_weight)
        self.feature_names_in_ = self.mid_.feature_names_in_
        self.n_features_in_ = self.mid_.n_features_in_
        return self

    def predict(self, X):
        out = np.column_stack([self.low_.predict(X), self.mid_.predict(X), self.high_.predict(X)])
        return np.sort(out, axis=1)          # low <= expected <= high, always

    def score(self, X, y, sample_weight=None):   # R2 of the expected value (scikit-learn's default score)
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X)[:, 1], sample_weight=sample_weight)
