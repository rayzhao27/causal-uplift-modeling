"""Seeded randomized promotion experiment with a known treatment effect.

The public Criteo Uplift file was reachable, but the transfer stalled twice and
the host rejected resume requests. This generator is the fallback: a 120,000-user
randomized experiment whose individual treatment effects are stored and then
withheld from every model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import EXPERIMENT_PARQUET, FEATURE_COLUMNS, N_USERS, SEED


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def conversion_cate(frame: pd.DataFrame) -> np.ndarray:
    """Known probability-scale treatment effect on conversion.

    The effect is larger for recent, engaged, newer customers who are not
    already loyal, and it is negative for disengaged discount-seekers.
    """
    recent = np.exp(-frame["recency_days"].to_numpy() / 21.0)
    engaged = np.tanh(frame["sessions_30d"].to_numpy() / 4.0)
    new_customer = np.exp(-frame["tenure_months"].to_numpy() / 10.0)
    purchases = frame["prior_purchases"].to_numpy()
    moderate_buyer = np.exp(-0.35 * (purchases - 2.0) ** 2)
    sleeping_dog = (
        (frame["discount_sensitivity"].to_numpy() > 0.75) & (frame["sessions_30d"].to_numpy() <= 1)
    ).astype(float)
    return (
        -0.012
        + 0.055 * engaged * recent
        + 0.028 * moderate_buyer
        + 0.022 * new_customer * engaged
        - 0.018 * frame["is_loyalty_member"].to_numpy()
        - 0.010 * frame["device_mobile"].to_numpy()
        + 0.008 * frame["device_desktop"].to_numpy()
        - 0.020 * sleeping_dog
    )


def visit_cate(frame: pd.DataFrame) -> np.ndarray:
    """Known probability-scale treatment effect on a site visit."""
    recent = np.exp(-frame["recency_days"].to_numpy() / 21.0)
    engaged = np.tanh(frame["sessions_30d"].to_numpy() / 4.0)
    return (
        0.01
        + 0.07 * engaged * recent
        + 0.02 * frame["device_desktop"].to_numpy()
        - 0.015 * frame["is_loyalty_member"].to_numpy()
    )


def _fit_effect(baseline: np.ndarray, effect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clip the effect so both potential outcomes stay inside (0.001, 0.95)."""
    tau = np.minimum(effect, 0.95 - baseline)
    tau = np.maximum(tau, 0.001 - baseline)
    return baseline, tau


def generate_experiment(n: int = N_USERS, seed: int = SEED) -> pd.DataFrame:
    """Draw one randomized experiment.

    Args:
        n: Number of users.
        seed: Generator seed.

    Returns:
        One row per user, including true conditional treatment effects.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    age = rng.integers(18, 71, size=n)
    tenure = np.clip(rng.gamma(2.2, 8.0, size=n), 0, 120)
    sessions = rng.poisson(np.clip(np.exp(0.7 + 0.45 * latent), 0.2, 12))
    email_opens = rng.poisson(np.clip(np.exp(0.15 + 0.35 * latent), 0.15, 8))
    pages = rng.poisson(np.clip(np.exp(1.3 + 0.40 * latent), 0.4, 20))
    purchases = rng.poisson(np.clip(0.4 + 0.55 * sessions / 3 + 0.25 * latent.clip(-1, 2), 0.15, 6))
    spend = rng.lognormal(mean=3.1 + 0.15 * purchases, sigma=0.45)
    recency = np.clip(rng.exponential(28 + 8 * (sessions == 0)), 0, 240)
    order_value = np.clip(rng.normal(48 + 4 * (purchases > 0), 18, size=n), 8, 220)
    discount = np.clip(rng.beta(2.2, 3.4, size=n), 0, 1)
    loyalty = rng.binomial(1, 1 / (1 + np.exp(1.1 - 0.35 * purchases - 0.02 * tenure)))
    weekend = rng.binomial(1, 0.42, size=n)
    device = rng.choice(["mobile", "desktop", "tablet"], size=n, p=[0.55, 0.35, 0.10])
    frame = pd.DataFrame(
        {
            "row_id": np.arange(1, n + 1),
            "age": age.astype(float),
            "tenure_months": tenure,
            "prior_purchases": purchases.astype(float),
            "prior_spend": spend,
            "recency_days": recency,
            "sessions_30d": sessions.astype(float),
            "email_opens_30d": email_opens.astype(float),
            "pages_viewed_30d": pages.astype(float),
            "avg_order_value": order_value,
            "discount_sensitivity": discount,
            "is_loyalty_member": loyalty.astype(float),
            "is_weekend_shopper": weekend.astype(float),
            "device_mobile": (device == "mobile").astype(float),
            "device_desktop": (device == "desktop").astype(float),
        }
    )
    treatment = rng.binomial(1, 0.5, size=n).astype(float)
    # Exposure happens only after a user is assigned to treatment.
    exposure = np.where(treatment == 1, rng.binomial(1, 0.82, size=n), 0).astype(float)

    visit_logit = (
        -1.20
        + 0.11 * frame["sessions_30d"].to_numpy()
        - 0.008 * frame["recency_days"].to_numpy()
        + 0.35 * frame["is_loyalty_member"].to_numpy()
        + 0.12 * frame["device_desktop"].to_numpy()
        - 0.06 * frame["device_mobile"].to_numpy()
    )
    visit_base = _sigmoid(visit_logit)
    _, tau_visit = _fit_effect(visit_base, visit_cate(frame))
    visit = rng.binomial(1, visit_base + treatment * tau_visit).astype(float)

    conversion_logit = (
        -2.65
        + 0.08 * frame["prior_purchases"].to_numpy()
        + 0.09 * frame["sessions_30d"].to_numpy()
        + 0.55 * frame["is_loyalty_member"].to_numpy()
        - 0.010 * frame["recency_days"].to_numpy()
        + 0.035 * frame["email_opens_30d"].to_numpy()
        + 0.015 * frame["pages_viewed_30d"].to_numpy()
        + 0.12 * frame["device_desktop"].to_numpy()
        - 0.05 * frame["device_mobile"].to_numpy()
    )
    conversion_base = _sigmoid(conversion_logit)
    _, tau_conversion = _fit_effect(conversion_base, conversion_cate(frame))
    converted = rng.binomial(1, conversion_base + treatment * tau_conversion).astype(float)

    frame["treatment"] = treatment
    frame["exposure"] = exposure
    frame["visit"] = visit
    frame["conversion"] = converted
    frame["true_cate_visit"] = tau_visit
    frame["true_cate_conversion"] = tau_conversion
    ordered = ["row_id", "treatment", *FEATURE_COLUMNS, "exposure", "visit", "conversion", "true_cate_visit", "true_cate_conversion"]
    return frame[ordered]


def materialize_experiment(n: int = N_USERS, seed: int = SEED, force: bool = False) -> pd.DataFrame:
    """Generate the experiment and store it as parquet.

    Args:
        n: Number of users.
        seed: Generator seed.
        force: Regenerate even if a parquet file already exists.

    Returns:
        The experiment frame.
    """
    if EXPERIMENT_PARQUET.exists() and not force:
        frame = pd.read_parquet(EXPERIMENT_PARQUET)
        if len(frame) == n:
            return frame
    EXPERIMENT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_experiment(n=n, seed=seed)
    frame.to_parquet(EXPERIMENT_PARQUET, index=False)
    return frame
