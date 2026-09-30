"""Tests for inference, balance, uplift metrics, and estimator recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.causal.ate import aipw_ate, cross_fit_nuisance, difference_in_means, ipw_ate, weight_diagnostics
from src.stats.ab_test import wald_risk_difference
from src.stats.balance import smd_from_arrays, standardized_mean_difference
from src.uplift.metrics import qini_coefficient
from src.uplift.models import RLearner, TLearner
from src.uplift.policy import choose_fraction, lookup_fraction, policy_curve
from src.utils.config import covariate_columns


def test_wald_interval_matches_the_formula() -> None:
    p_t, p_c, diff, se, low, high = wald_risk_difference(120, 1000, 100, 1000)
    expected_se = np.sqrt(0.12 * 0.88 / 1000 + 0.10 * 0.90 / 1000)
    assert p_t == 0.12
    assert p_c == 0.10
    assert abs(diff - 0.02) < 1e-12
    assert abs(se - expected_se) < 1e-12
    assert low < diff < high


def test_standardized_mean_difference_scale() -> None:
    assert standardized_mean_difference(1.0, 0.0, 1.0, 1.0) == 1.0
    assert standardized_mean_difference(0.2, 0.2, 1.0, 2.0) == 0.0
    rng = np.random.default_rng(0)
    values = rng.normal(size=4000)
    treatment = np.array([0, 1] * 2000)
    assert abs(smd_from_arrays(values, treatment)) < 0.1


def test_covariates_exclude_post_treatment_fields() -> None:
    columns = covariate_columns()
    assert "device_mobile" in columns
    assert "recency_days" in columns
    assert len(columns) == 14
    for forbidden in (
        "exposure",
        "visit",
        "conversion",
        "treatment",
        "row_id",
        "true_cate_conversion",
        "true_cate_visit",
    ):
        assert forbidden not in columns


def _simulate(n: int = 8000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    treatment = rng.binomial(1, 0.5, size=n).astype(float)
    tau = np.where(x[:, 0] > 0, 0.12, -0.02)
    p0 = np.clip(0.20 + 0.04 * x[:, 1], 0.05, 0.50)
    p1 = np.clip(p0 + tau, 0.01, 0.95)
    probability = np.where(treatment == 1, p1, p0)
    outcome = rng.binomial(1, probability).astype(float)
    return x, outcome, treatment, p1 - p0


def test_aipw_and_ipw_recover_a_known_effect() -> None:
    x, y, t, tau = _simulate(n=10000, seed=1)
    propensity, mu0, mu1 = cross_fit_nuisance(x, y, t, n_splits=2, seed=1)
    assert propensity.min() > 0
    assert propensity.max() < 1
    diagnostics = weight_diagnostics(t, propensity)
    assert diagnostics["weight_min"] > 0
    assert np.isfinite(diagnostics["effective_sample_size"])
    dr = aipw_ate(y, t, propensity, mu0, mu1)
    weighted = ipw_ate(y, t, propensity)
    naive = difference_in_means(y, t)
    truth = float(tau.mean())
    assert abs(dr.estimate - truth) < 0.03
    assert abs(weighted.estimate - truth) < 0.03
    assert abs(naive.estimate - truth) < 0.03
    assert dr.ci_low < dr.estimate < dr.ci_high


def test_policy_at_full_coverage_matches_the_risk_difference() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    treatment = rng.binomial(1, 0.4, size=n).astype(float)
    outcome = rng.binomial(1, 0.1 + 0.05 * treatment).astype(float)
    score = rng.normal(size=n)
    curve = policy_curve(outcome, treatment, score, fractions=[1.0], min_arm=10)
    ate = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    assert abs(float(curve.iloc[0]["uplift_rate"]) - ate) < 1e-10
    assert int(curve.iloc[0]["users_targeted"]) == n


def test_qini_prefers_the_true_ranking() -> None:
    x, y, t, tau = _simulate(n=4000, seed=3)
    informed = qini_coefficient(y, t, tau, seed=3)
    rng = np.random.default_rng(3)
    random_score = qini_coefficient(y, t, rng.normal(size=len(y)), seed=3)
    assert informed > random_score


def test_choose_fraction_respects_profit_and_the_do_nothing_option() -> None:
    curve = pd.DataFrame(
        {
            "target_fraction": [0.5, 1.0],
            "users_targeted": [50, 100],
            "incremental_conversions": [20.0, 10.0],
            "uplift_rate": [0.4, 0.1],
            "share_of_treat_all_incremental": [2.0, 1.0],
        }
    )
    assert choose_fraction(curve, value=10, cost=0, grid=(0.5, 1.0)) == 0.5
    assert choose_fraction(curve, value=1, cost=10, grid=(0.5, 1.0)) == 0.0
    nobody = lookup_fraction(curve, 0.0)
    assert nobody["users_targeted"] == 0
    assert nobody["incremental_conversions"] == 0


def test_uplift_models_rank_a_known_treatment_effect() -> None:
    x, y, t, tau = _simulate(n=5000, seed=4)
    learner = TLearner(seed=4, max_iter=25, min_samples_leaf=40)
    learner.fit(x[:3500], y[:3500], t[:3500])
    predicted = learner.predict(x[3500:])
    assert roc_auc_score((tau[3500:] > np.median(tau[3500:])).astype(int), predicted) > 0.65
    r_learner = RLearner(seed=4, n_splits=2, max_iter=25, min_samples_leaf=40)
    r_learner.fit(x[:3500], y[:3500], t[:3500])
    r_pred = r_learner.predict(x[3500:])
    assert np.corrcoef(r_pred, tau[3500:])[0, 1] > 0.15


def test_synthetic_experiment_recovers_its_known_average_effect() -> None:
    from src.data.synthetic import generate_experiment

    frame = generate_experiment(n=20000, seed=7)
    treated = frame["treatment"] == 1
    observed = frame.loc[treated, "conversion"].mean() - frame.loc[~treated, "conversion"].mean()
    truth = frame["true_cate_conversion"].mean()
    assert 0.04 < frame["conversion"].mean() < 0.25
    assert abs(observed - truth) < 0.02
    assert set(covariate_columns()).isdisjoint({"true_cate_conversion", "true_cate_visit", "exposure"})
