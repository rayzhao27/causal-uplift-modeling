"""Intent-to-treat inference for a binary outcome and a binary treatment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)


@dataclass(frozen=True)
class ProportionEffect:
    """Unadjusted comparison of two independent proportions."""

    outcome: str
    n_treatment: int
    n_control: int
    conversions_treatment: int
    conversions_control: int
    rate_treatment: float
    rate_control: float
    absolute_lift: float
    absolute_se: float
    ci_low: float
    ci_high: float
    relative_lift: float
    relative_lift_ci_low: float
    relative_lift_ci_high: float
    risk_ratio: float
    risk_ratio_ci_low: float
    risk_ratio_ci_high: float
    z_statistic: float
    z_pvalue: float
    chi2_statistic: float
    chi2_pvalue: float
    rate_treatment_ci_low: float
    rate_treatment_ci_high: float
    rate_control_ci_low: float
    rate_control_ci_high: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def wald_risk_difference(
    successes_t: int,
    n_t: int,
    successes_c: int,
    n_c: int,
    alpha: float = 0.05,
) -> tuple[float, float, float, float, float, float]:
    """Wald estimate and interval for a difference in proportions.

    Args:
        successes_t: Outcome count in treatment.
        n_t: Treatment sample size.
        successes_c: Outcome count in control.
        n_c: Control sample size.
        alpha: Two-sided error rate.

    Returns:
        Treatment rate, control rate, difference, standard error, CI low, CI high.
    """
    if min(n_t, n_c) <= 0:
        raise ValueError("Both arms need a positive sample size.")
    p_t = successes_t / n_t
    p_c = successes_c / n_c
    diff = p_t - p_c
    se = float(np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))
    z = float(stats.norm.ppf(1 - alpha / 2))
    return float(p_t), float(p_c), float(diff), se, float(diff - z * se), float(diff + z * se)


def risk_ratio_interval(
    p_t: float,
    n_t: int,
    p_c: float,
    n_c: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Delta-method confidence interval for the risk ratio on the log scale.

    Returns:
        Risk ratio, CI low, CI high.
    """
    if p_t <= 0 or p_c <= 0:
        raise ValueError("Risk ratio is undefined when either rate is zero.")
    rr = p_t / p_c
    se_log = np.sqrt((1 - p_t) / (n_t * p_t) + (1 - p_c) / (n_c * p_c))
    z = float(stats.norm.ppf(1 - alpha / 2))
    return float(rr), float(np.exp(np.log(rr) - z * se_log)), float(np.exp(np.log(rr) + z * se_log))


def two_proportion_inference(
    successes_t: int,
    n_t: int,
    successes_c: int,
    n_c: int,
    outcome: str,
    alpha: float = 0.05,
) -> ProportionEffect:
    """Z-test, chi-square test, and Wald interval for an intent-to-treat contrast.

    Args:
        successes_t: Outcomes in the treatment arm.
        n_t: Treatment arm size.
        successes_c: Outcomes in the control arm.
        n_c: Control arm size.
        outcome: Outcome name stored on the result.
        alpha: Confidence level complement.

    Returns:
        Effect sizes, tests, and intervals.
    """
    p_t, p_c, diff, se, ci_low, ci_high = wald_risk_difference(successes_t, n_t, successes_c, n_c, alpha)
    sm_low, sm_high = confint_proportions_2indep(
        successes_t,
        n_t,
        successes_c,
        n_c,
        compare="diff",
        method="wald",
        alpha=alpha,
        correction=False,
    )
    if abs(ci_low - float(sm_low)) > 1e-8 or abs(ci_high - float(sm_high)) > 1e-8:
        raise RuntimeError("Internal Wald interval does not match statsmodels.")

    z_stat, z_p = proportions_ztest(
        np.array([successes_t, successes_c]),
        np.array([n_t, n_c]),
        alternative="two-sided",
        prop_var=False,
    )
    table = np.array(
        [
            [successes_t, n_t - successes_t],
            [successes_c, n_c - successes_c],
        ]
    )
    chi2, chi_p, _, _ = stats.chi2_contingency(table, correction=False)
    rr, rr_low, rr_high = risk_ratio_interval(p_t, n_t, p_c, n_c, alpha)
    t_low, t_high = proportion_confint(successes_t, n_t, alpha=alpha, method="wilson")
    c_low, c_high = proportion_confint(successes_c, n_c, alpha=alpha, method="wilson")
    return ProportionEffect(
        outcome=outcome,
        n_treatment=int(n_t),
        n_control=int(n_c),
        conversions_treatment=int(successes_t),
        conversions_control=int(successes_c),
        rate_treatment=p_t,
        rate_control=p_c,
        absolute_lift=diff,
        absolute_se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        relative_lift=float(rr - 1),
        relative_lift_ci_low=float(rr_low - 1),
        relative_lift_ci_high=float(rr_high - 1),
        risk_ratio=rr,
        risk_ratio_ci_low=rr_low,
        risk_ratio_ci_high=rr_high,
        z_statistic=float(z_stat),
        z_pvalue=float(z_p),
        chi2_statistic=float(chi2),
        chi2_pvalue=float(chi_p),
        rate_treatment_ci_low=float(t_low),
        rate_treatment_ci_high=float(t_high),
        rate_control_ci_low=float(c_low),
        rate_control_ci_high=float(c_high),
    )


def cohens_h(p1: float, p0: float) -> float:
    """Cohen's h for two proportions."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p0)))


def achieved_power(p_t: float, p_c: float, n_t: int, n_c: int, alpha: float = 0.05) -> float:
    """Two-sided achieved power of the observed risk difference."""
    effect = abs(cohens_h(p_t, p_c))
    if effect == 0:
        return float(alpha)
    power = NormalIndPower().power(
        effect_size=effect,
        nobs1=n_t,
        alpha=alpha,
        ratio=n_c / n_t,
        alternative="two-sided",
    )
    return float(power)


def minimum_detectable_effect(
    p_c: float,
    n_t: int,
    n_c: int,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
    """Absolute risk difference detectable at the requested power.

    Uses the design standard error at the control rate. This is the usual
    experiment-planning approximation and is on the probability scale.

    Args:
        p_c: Baseline (control) rate.
        n_t: Treatment sample size.
        n_c: Control sample size.
        power: Target power.
        alpha: Two-sided size.

    Returns:
        Minimum detectable absolute lift.
    """
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_beta = float(stats.norm.ppf(power))
    se = np.sqrt(p_c * (1 - p_c) * (1 / n_t + 1 / n_c))
    return float((z_alpha + z_beta) * se)


def required_sample_size(
    p_c: float,
    delta: float,
    allocation_ratio_c_over_t: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Sample size for an absolute lift, at the experiment's allocation.

    Args:
        p_c: Baseline rate.
        delta: Absolute lift to detect (treatment minus control).
        allocation_ratio_c_over_t: Control N divided by treatment N.
        power: Target power.
        alpha: Two-sided size.

    Returns:
        Required treatment, control, and total sample sizes.
    """
    p_t = min(max(p_c + delta, 1e-8), 1 - 1e-8)
    effect = abs(cohens_h(p_t, p_c))
    n_t = float(
        NormalIndPower().solve_power(
            effect_size=effect,
            power=power,
            alpha=alpha,
            ratio=allocation_ratio_c_over_t,
            alternative="two-sided",
        )
    )
    n_c = n_t * allocation_ratio_c_over_t
    return {"delta": delta, "n_treatment": n_t, "n_control": n_c, "n_total": n_t + n_c}


def power_report(effect: ProportionEffect, deltas: list[float]) -> dict[str, object]:
    """Achieved power, MDE, and required sample sizes for a grid of lifts."""
    power = achieved_power(effect.rate_treatment, effect.rate_control, effect.n_treatment, effect.n_control)
    mde = minimum_detectable_effect(effect.rate_control, effect.n_treatment, effect.n_control)
    ratio = effect.n_control / effect.n_treatment
    sizes = [
        required_sample_size(effect.rate_control, delta, ratio)
        for delta in deltas
    ]
    return {
        "outcome": effect.outcome,
        "observed_effect": effect.absolute_lift,
        "cohens_h": cohens_h(effect.rate_treatment, effect.rate_control),
        "achieved_power": power,
        "mde_80_power": mde,
        "required_sample_sizes": sizes,
    }


def experiment_table(effect: ProportionEffect, power: dict[str, object]) -> pd.DataFrame:
    """One-row display table for the README and dashboard."""
    row = {
        "outcome": effect.outcome,
        "n_treatment": effect.n_treatment,
        "n_control": effect.n_control,
        "rate_treatment": effect.rate_treatment,
        "rate_control": effect.rate_control,
        "absolute_lift": effect.absolute_lift,
        "ci_low": effect.ci_low,
        "ci_high": effect.ci_high,
        "relative_lift": effect.relative_lift,
        "risk_ratio": effect.risk_ratio,
        "z_statistic": effect.z_statistic,
        "p_value": effect.z_pvalue,
        "chi2_statistic": effect.chi2_statistic,
        "chi2_pvalue": effect.chi2_pvalue,
        "achieved_power": power["achieved_power"],
        "mde_80_power": power["mde_80_power"],
    }
    return pd.DataFrame([row])
