"""Refutation checks that do not add a modeling dependency.

Placebo treatment, an irrelevant covariate, the bootstrap, and subsample
repeats all ask the same question: does the estimated effect behave like a
real contrast in a randomized experiment, or like a fitting artifact?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.ate import difference_in_means, regression_adjusted_ate


def placebo_treatment(
    y: np.ndarray,
    t: np.ndarray,
    n_perm: int = 200,
    seed: int = 42,
) -> dict[str, float | np.ndarray]:
    """Permute treatment and recompute the difference in means.

    Args:
        y: Outcome. Held fixed.
        t: Original treatment. Only its values are reshuffled.
        n_perm: Number of permutations.
        seed: Permutation seed.

    Returns:
        Observed effect, permutation distribution, and a two-sided permutation p-value.
    """
    observed = difference_in_means(y, t).estimate
    rng = np.random.default_rng(seed)
    placebo = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(t)
        placebo[i] = difference_in_means(y, shuffled).estimate
    p_value = float((np.sum(np.abs(placebo) >= abs(observed)) + 1) / (n_perm + 1))
    return {
        "observed_ate": float(observed),
        "placebo_mean": float(placebo.mean()),
        "placebo_std": float(placebo.std(ddof=1)),
        "placebo_p_value": p_value,
        "placebo_distribution": placebo,
    }


def random_common_cause(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    seed: int = 42,
) -> dict[str, float]:
    """Add an independent noise covariate and refit regression adjustment.

    A real treatment contrast should barely move. A large move means the
    adjusted estimator is unstable.

    Args:
        x: Pre-treatment covariates.
        y: Outcome.
        t: Treatment.
        seed: Noise seed.

    Returns:
        Baseline and augmented treatment coefficients.
    """
    baseline = regression_adjusted_ate(x, y, t)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(len(y), 1))
    augmented = regression_adjusted_ate(
        np.column_stack([x, noise]),
        y,
        t,
        name="Regression with noise covariate",
        notes="Sensitivity check.",
    )
    return {
        "baseline_ate": baseline.estimate,
        "augmented_ate": augmented.estimate,
        "absolute_change": abs(augmented.estimate - baseline.estimate),
    }


def bootstrap_difference_in_means(
    y: np.ndarray,
    t: np.ndarray,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, float | np.ndarray]:
    """Nonparametric bootstrap of the unadjusted risk difference.

    Args:
        y: Outcome.
        t: Treatment.
        n_boot: Bootstrap replications.
        seed: Resampling seed.

    Returns:
        Bootstrap distribution and percentile interval.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        estimates[i] = difference_in_means(y[idx], t[idx]).estimate
    return {
        "boot_mean": float(estimates.mean()),
        "boot_se": float(estimates.std(ddof=1)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "distribution": estimates,
    }


def subsample_stability(
    y: np.ndarray,
    t: np.ndarray,
    n_reps: int = 30,
    fraction: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Repeat the unadjusted contrast on random subsets.

    Args:
        y: Outcome.
        t: Treatment.
        n_reps: Number of subsets.
        fraction: Subset size as a share of the sample.
        seed: Subsampling seed.

    Returns:
        One row per subset.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    size = int(round(fraction * n))
    rows = []
    for i in range(n_reps):
        idx = rng.choice(n, size=size, replace=False)
        estimate = difference_in_means(y[idx], t[idx])
        rows.append({"rep": i, "estimate": estimate.estimate, "ci_low": estimate.ci_low, "ci_high": estimate.ci_high})
    return pd.DataFrame(rows)
