"""Uplift models fit only on the training split.

Predicted values are individual treatment effects on the probability scale:
P(Y=1 | X, treatment=1) − P(Y=1 | X, treatment=0), or the R-learner analogue.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.causal.ate import _fit_binary_probabilities, apply_winsor, clip_propensity, winsor_bounds


def _classifier(seed: int, rare: bool) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=80 if rare else 120,
        learning_rate=0.08,
        max_depth=3 if rare else 4,
        min_samples_leaf=800 if rare else 150,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=8,
        random_state=seed,
    )


def _regressor(seed: int, rare: bool) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=80 if rare else 120,
        learning_rate=0.08,
        max_depth=3 if rare else 4,
        min_samples_leaf=800 if rare else 150,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=8,
        random_state=seed,
    )


class TLearner:
    """Two outcome models, one fit on each treatment arm."""

    def __init__(self, seed: int = 42, max_iter: int | None = None, min_samples_leaf: int | None = None) -> None:
        self.seed = seed
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.model_treated: HistGradientBoostingClassifier | None = None
        self.model_control: HistGradientBoostingClassifier | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> "TLearner":
        """Fit arm-specific outcome models.

        Args:
            x: Training covariates.
            y: Training outcome.
            t: Training treatment.

        Returns:
            The fitted learner.
        """
        rare = bool(np.mean(y) < 0.02)
        treated = t.astype(bool)
        self.model_treated = _classifier(self.seed, rare)
        self.model_control = _classifier(self.seed + 1, rare)
        if self.max_iter is not None:
            self.model_treated.set_params(max_iter=self.max_iter, early_stopping=False)
            self.model_control.set_params(max_iter=self.max_iter, early_stopping=False)
        if self.min_samples_leaf is not None:
            self.model_treated.set_params(min_samples_leaf=self.min_samples_leaf)
            self.model_control.set_params(min_samples_leaf=self.min_samples_leaf)
        self.model_treated.fit(x[treated], y[treated].astype(int))
        self.model_control.fit(x[~treated], y[~treated].astype(int))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicted treatment effect on the probability scale."""
        if self.model_treated is None or self.model_control is None:
            raise RuntimeError("TLearner must be fit before predict.")
        return self.model_treated.predict_proba(x)[:, 1] - self.model_control.predict_proba(x)[:, 1]


class RLearner:
    """Robinson's R-learner with a gradient-boosted second stage.

    Nuisance functions are cross-fit logistic regressions. The second stage is
    a tree model of the residualized outcome, which is the heterogeneous
    treatment effect.
    """

    def __init__(self, seed: int = 42, n_splits: int = 3, max_iter: int | None = None, min_samples_leaf: int | None = None) -> None:
        self.seed = seed
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.model: HistGradientBoostingRegressor | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> "RLearner":
        """Cross-fit nuisances on the training rows, then fit the CATE model.

        Args:
            x: Training covariates.
            y: Training outcome.
            t: Training treatment.

        Returns:
            The fitted learner.
        """
        n = x.shape[0]
        propensity = np.zeros(n)
        mu = np.zeros(n)
        lo, hi = winsor_bounds(x)
        x_nuisance = apply_winsor(x, lo, hi)
        splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for train_idx, test_idx in splitter.split(x_nuisance):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_nuisance[train_idx])
            x_test = scaler.transform(x_nuisance[test_idx])
            t_train = t[train_idx].astype(int)
            propensity[test_idx] = _fit_binary_probabilities(x_train, t_train, x_test)
            mu[test_idx] = _fit_binary_probabilities(x_train, y[train_idx], x_test)
        e = clip_propensity(propensity, (0.05, 0.95))
        pseudo = (y - mu) / (t - e)
        weights = (t - e) ** 2
        rare = bool(np.mean(y) < 0.02)
        self.model = _regressor(self.seed, rare)
        if self.max_iter is not None:
            self.model.set_params(max_iter=self.max_iter, early_stopping=False)
        if self.min_samples_leaf is not None:
            self.model.set_params(min_samples_leaf=self.min_samples_leaf)
        self.model.fit(x, pseudo, sample_weight=weights)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicted conditional average treatment effect."""
        if self.model is None:
            raise RuntimeError("RLearner must be fit before predict.")
        return self.model.predict(x)
