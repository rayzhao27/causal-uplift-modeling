"""Average treatment effect estimators on the probability scale.

The target is the intent-to-treat risk difference. In this randomized experiment
the propensity score should be nearly constant, so the unadjusted difference,
regression adjustment, IPW, and AIPW are expected to agree. Disagreement would
indicate a bug, propensity overfitting, or a post-treatment covariate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.utils.config import N_FOLDS, PROPENSITY_CLIP, SEED, WINSOR_QUANTILE


@dataclass(frozen=True)
class ATEEstimate:
    """A point estimate of the risk difference with a normal interval."""

    name: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    causal: bool = True
    notes: str = ""

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


def _interval(estimate: float, se: float, alpha_z: float = 1.959963984540054) -> tuple[float, float]:
    return float(estimate - alpha_z * se), float(estimate + alpha_z * se)


def difference_in_means(y: np.ndarray, t: np.ndarray) -> ATEEstimate:
    """Unadjusted difference in mean outcomes.

    Args:
        y: Binary outcome.
        t: Binary treatment.

    Returns:
        Wald estimate of the risk difference.
    """
    treated = t.astype(bool)
    y1 = y[treated]
    y0 = y[~treated]
    p1 = float(y1.mean())
    p0 = float(y0.mean())
    estimate = p1 - p0
    se = float(np.sqrt(p1 * (1 - p1) / len(y1) + p0 * (1 - p0) / len(y0)))
    low, high = _interval(estimate, se)
    return ATEEstimate("Sample difference in means", estimate, se, low, high, int(len(y)), notes="Unadjusted ITT on this sample.")


def winsor_bounds(x: np.ndarray, quantile: float = WINSOR_QUANTILE) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise clip bounds. Uses covariates only."""
    lo = np.quantile(x, quantile, axis=0)
    hi = np.quantile(x, 1 - quantile, axis=0)
    return lo, hi


def apply_winsor(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Clip columns to precomputed bounds."""
    return np.clip(x, lo, hi)


def regression_adjusted_ate(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    name: str = "Regression adjustment",
    causal: bool = True,
    notes: str = "Linear probability model with HC1 standard errors. Covariates are pre-treatment and winsorized.",
) -> ATEEstimate:
    """OLS coefficient on treatment, adjusting for covariates.

    The linear probability model estimates the risk difference directly, which
    keeps this estimator on the same scale as the experiment contrast.

    Args:
        x: Pre-treatment covariate matrix.
        y: Outcome.
        t: Treatment.
        name: Label stored on the result.
        causal: False when the specification is deliberately biased.
        notes: Interpretation note.

    Returns:
        Treatment coefficient and robust confidence interval.
    """
    lo, hi = winsor_bounds(x)
    clipped = apply_winsor(x, lo, hi)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clipped)
    design = sm.add_constant(np.column_stack([t.astype(float), scaled]), has_constant="add")
    fit = sm.OLS(y.astype(float), design).fit(cov_type="HC1")
    estimate = float(fit.params[1])
    se = float(fit.bse[1])
    low, high = _interval(estimate, se)
    return ATEEstimate(name, estimate, se, low, high, int(len(y)), causal=causal, notes=notes)


def _logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=400, solver="lbfgs")


def _fit_binary_probabilities(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Predict class-1 probabilities, falling back to the training base rate."""
    if np.unique(y_train).size < 2:
        return np.full(x_test.shape[0], float(y_train.mean()))
    model = _logistic()
    model.fit(x_train, y_train.astype(int))
    return model.predict_proba(x_test)[:, 1]


def cross_fit_nuisance(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_splits: int = N_FOLDS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold propensity and arm-specific outcome regressions.

    Args:
        x: Covariate matrix.
        y: Outcome.
        t: Treatment.
        n_splits: Number of cross-fitting folds.
        seed: Fold shuffle seed.

    Returns:
        Propensity, mu0, and mu1, each aligned to ``x``.
    """
    n = x.shape[0]
    propensity = np.zeros(n)
    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    lo, hi = winsor_bounds(x)
    clipped = apply_winsor(x, lo, hi)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(clipped):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(clipped[train_idx])
        x_test = scaler.transform(clipped[test_idx])
        t_train = t[train_idx].astype(int)
        propensity[test_idx] = _fit_binary_probabilities(x_train, t_train, x_test)
        for arm, store in ((1, mu1), (0, mu0)):
            mask = t_train == arm
            store[test_idx] = _fit_binary_probabilities(x_train[mask], y[train_idx][mask], x_test)
    return propensity, mu0, mu1


def clip_propensity(propensity: np.ndarray, bounds: tuple[float, float] = PROPENSITY_CLIP) -> np.ndarray:
    """Clip propensity scores into ``bounds``."""
    return np.clip(propensity, bounds[0], bounds[1])


def stabilized_weights(t: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    """Stabilized IPW weights with mean near one in a correctly specified model."""
    p_hat = float(np.mean(t))
    e = clip_propensity(propensity)
    return np.where(t == 1, p_hat / e, (1 - p_hat) / (1 - e))


def weight_diagnostics(t: np.ndarray, propensity: np.ndarray) -> dict[str, float]:
    """Summaries used to decide whether IPW weights are trustworthy."""
    raw = propensity.astype(float)
    clipped = clip_propensity(raw)
    weights = stabilized_weights(t, raw)
    ess = float((weights.sum() ** 2) / np.sum(weights**2))
    return {
        "propensity_min": float(raw.min()),
        "propensity_p01": float(np.quantile(raw, 0.01)),
        "propensity_mean": float(raw.mean()),
        "propensity_p99": float(np.quantile(raw, 0.99)),
        "propensity_max": float(raw.max()),
        "share_clipped": float(np.mean(raw != clipped)),
        "weight_min": float(weights.min()),
        "weight_p50": float(np.quantile(weights, 0.50)),
        "weight_p95": float(np.quantile(weights, 0.95)),
        "weight_p99": float(np.quantile(weights, 0.99)),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "effective_sample_size": ess,
        "effective_sample_fraction": ess / len(weights),
    }


def ipw_ate(y: np.ndarray, t: np.ndarray, propensity: np.ndarray) -> ATEEstimate:
    """Hájek inverse-probability-weighted risk difference.

    Propensity scores should be cross-fit. The standard error treats those
    scores as fixed, which is slightly conservative when the propensity model
    is estimated.

    Args:
        y: Outcome.
        t: Treatment.
        propensity: P(treatment = 1 | covariates), one score per row.

    Returns:
        Weighted risk difference and interval.
    """
    e = clip_propensity(propensity)
    w1 = t / e
    w0 = (1 - t) / (1 - e)
    mu1 = float(np.sum(w1 * y) / np.sum(w1))
    mu0 = float(np.sum(w0 * y) / np.sum(w0))
    estimate = mu1 - mu0
    phi = (w1 / np.mean(w1)) * (y - mu1) - (w0 / np.mean(w0)) * (y - mu0)
    se = float(np.std(phi, ddof=1) / np.sqrt(len(y)))
    low, high = _interval(estimate, se)
    return ATEEstimate(
        "IPW (Hájek)",
        estimate,
        se,
        low,
        high,
        int(len(y)),
        notes="Cross-fit propensity, scores clipped to [0.01, 0.99]. SE treats scores as fixed.",
    )


def aipw_ate(
    y: np.ndarray,
    t: np.ndarray,
    propensity: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> ATEEstimate:
    """Augmented IPW / doubly robust risk difference from cross-fit nuisances.

    The estimate is consistent if either the propensity model or both outcome
    regressions are consistent. Cross-fitting keeps the nuisance fits from
    overfitting the same rows used in the final average.

    Args:
        y: Outcome.
        t: Treatment.
        propensity: Cross-fit propensity.
        mu0: Cross-fit E[Y | X, treatment = 0].
        mu1: Cross-fit E[Y | X, treatment = 1].

    Returns:
        Doubly robust estimate and influence-function interval.
    """
    e = clip_propensity(propensity)
    score = mu1 - mu0 + (t / e) * (y - mu1) - ((1 - t) / (1 - e)) * (y - mu0)
    estimate = float(np.mean(score))
    se = float(np.std(score, ddof=1) / np.sqrt(len(y)))
    low, high = _interval(estimate, se)
    return ATEEstimate(
        "AIPW (doubly robust)",
        estimate,
        se,
        low,
        high,
        int(len(y)),
        notes="Cross-fit logistic propensity and outcome models. Interval from the influence function.",
    )


def full_sample_itt(successes_t: int, n_t: int, successes_c: int, n_c: int) -> ATEEstimate:
    """Unadjusted ITT computed from arm counts, typically the full experiment."""
    p1 = successes_t / n_t
    p0 = successes_c / n_c
    estimate = float(p1 - p0)
    se = float(np.sqrt(p1 * (1 - p1) / n_t + p0 * (1 - p0) / n_c))
    low, high = _interval(estimate, se)
    return ATEEstimate(
        "Full-experiment difference in means",
        estimate,
        se,
        low,
        high,
        int(n_t + n_c),
        notes="Primary causal benchmark. Uses every randomized user, not the modeling sample.",
    )
