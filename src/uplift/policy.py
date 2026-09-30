"""Targeting-policy value from a randomized holdout.

Users are ranked by a model fit on other rows. Inside each prefix of that
ranking, treatment was still random, so the difference in outcome rates is an
unbiased estimate of the incremental effect of treating that slice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.uplift.metrics import ranking_order


def policy_curve(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    fractions: tuple[float, ...] | list[float],
    seed: int = 42,
    min_arm: int = 30,
) -> pd.DataFrame:
    """Empirical value of treating the top fraction of a ranked list.

    Args:
        y: Evaluation outcomes.
        t: Evaluation treatment assignment.
        score: Predicted uplift used only to rank.
        fractions: Shares of the evaluation sample to target, in (0, 1].
        seed: Tie-break seed. Must match the Qini ranking seed to be comparable.
        min_arm: Minimum treated and control counts required to report a contrast.

    Returns:
        One row per fraction with rates, incremental outcomes, and sampling error.
    """
    order = ranking_order(score, seed)
    y_sorted = y[order].astype(float)
    t_sorted = t[order].astype(float)
    score_sorted = score[order].astype(float)
    n = len(y)
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1.0 - t_sorted)
    cum_yt = np.cumsum(y_sorted * t_sorted)
    cum_yc = np.cumsum(y_sorted * (1.0 - t_sorted))
    cum_score = np.cumsum(score_sorted)
    rows: list[dict[str, float]] = []
    for fraction in fractions:
        k = int(round(float(fraction) * n))
        k = min(max(k, 1), n)
        nt = float(cum_t[k - 1])
        nc = float(cum_c[k - 1])
        if nt < min_arm or nc < min_arm:
            uplift = np.nan
            se = np.nan
            p1 = np.nan
            p0 = np.nan
        else:
            p1 = float(cum_yt[k - 1] / nt)
            p0 = float(cum_yc[k - 1] / nc)
            uplift = p1 - p0
            se = float(np.sqrt(p1 * (1 - p1) / nt + p0 * (1 - p0) / nc))
        incremental = uplift * k if np.isfinite(uplift) else np.nan
        rows.append(
            {
                "target_fraction": float(fraction),
                "users_targeted": k,
                "n_treatment_in_slice": nt,
                "n_control_in_slice": nc,
                "rate_treatment": p1,
                "rate_control": p0,
                "uplift_rate": uplift,
                "uplift_se": se,
                "incremental_conversions": incremental,
                "incremental_conversions_se": se * k if np.isfinite(se) else np.nan,
                "incremental_conversions_per_1000": uplift * 1000 if np.isfinite(uplift) else np.nan,
                "mean_predicted_cate": float(cum_score[k - 1] / k),
            }
        )
    frame = pd.DataFrame(rows)
    total = frame.loc[np.isclose(frame["target_fraction"], 1.0), "incremental_conversions"]
    total_value = float(total.iloc[0]) if len(total) else np.nan
    if np.isfinite(total_value) and abs(total_value) > 1e-8:
        frame["share_of_treat_all_incremental"] = frame["incremental_conversions"] / total_value
    else:
        frame["share_of_treat_all_incremental"] = np.nan
    return frame


def profit_of_row(row: pd.Series, value: float, cost: float) -> float:
    """Incremental profit of one policy row under explicit unit economics."""
    incremental = float(row["incremental_conversions"])
    users = float(row["users_targeted"])
    if not np.isfinite(incremental):
        return float("nan")
    return incremental * value - users * cost


def choose_fraction(curve: pd.DataFrame, value: float, cost: float, grid: tuple[float, ...] | None = None) -> float:
    """Pick the targeting fraction with the highest empirical profit.

    Treating nobody has profit zero and is chosen when every listed policy
    loses money. The search grid should be coarse (for example 5 points)
    so the rule is not fit to a one-point spike.

    Args:
        curve: Policy curve that contains ``grid``.
        value: Value of one incremental outcome.
        cost: Cost of treating one user.
        grid: Fractions eligible for selection. Defaults to every row.

    Returns:
        Selected fraction in ``grid``, or 0 when the best profit is not positive.
    """
    eligible = curve if grid is None else curve[curve["target_fraction"].round(2).isin(np.round(grid, 2))]
    profits = eligible.apply(lambda row: profit_of_row(row, value, cost), axis=1)
    if profits.isna().all() or float(profits.max()) <= 0:
        return 0.0
    return float(eligible.loc[profits.idxmax(), "target_fraction"])


def lookup_fraction(curve: pd.DataFrame, fraction: float) -> pd.Series:
    """Return the policy row closest to ``fraction``.

    A fraction of 0 means treat nobody and is synthesized rather than looked up.
    """
    if fraction <= 0:
        template = curve.iloc[0]
        data = {col: 0.0 for col in template.index}
        data["target_fraction"] = 0.0
        data["share_of_treat_all_incremental"] = 0.0
        return pd.Series(data)
    idx = (curve["target_fraction"] - fraction).abs().idxmin()
    return curve.loc[idx]


def scale_to_cohort(row: pd.Series, cohort_n: int, value: float, cost: float) -> dict[str, float]:
    """Project a holdout rate onto a future cohort of ``cohort_n`` similar users.

    The projection multiplies the holdout uplift rate by the number of users
    the policy would treat. It is not a second experiment.
    """
    fraction = float(row["target_fraction"])
    uplift = float(row["uplift_rate"]) if fraction > 0 else 0.0
    users = int(round(fraction * cohort_n))
    incremental = uplift * users
    return {
        "cohort_n": cohort_n,
        "target_fraction": fraction,
        "users_targeted": users,
        "incremental_outcomes": incremental,
        "incremental_outcomes_per_1000": uplift * 1000,
        "share_of_treat_all_incremental": float(row.get("share_of_treat_all_incremental", np.nan)),
        "campaign_cost": users * cost,
        "incremental_value": incremental * value,
        "incremental_profit": incremental * value - users * cost,
    }
