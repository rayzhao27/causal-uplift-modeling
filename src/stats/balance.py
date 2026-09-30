"""Covariate balance and data-quality summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import FEATURE_COLUMNS, SMD_BALANCE_THRESHOLD


def standardized_mean_difference(mean_t: float, mean_c: float, std_t: float, std_c: float) -> float:
    """Pooled-SD standardized difference, treatment minus control.

    Args:
        mean_t: Treatment-arm mean.
        mean_c: Control-arm mean.
        std_t: Treatment-arm sample standard deviation.
        std_c: Control-arm sample standard deviation.

    Returns:
        Standardized mean difference. Zero when both arms have zero variance.
    """
    pooled = np.sqrt((std_t**2 + std_c**2) / 2.0)
    if pooled == 0 or not np.isfinite(pooled):
        return 0.0
    return float((mean_t - mean_c) / pooled)


def smd_from_arrays(values: np.ndarray, treatment: np.ndarray) -> float:
    """Standardized mean difference from microdata."""
    treated = treatment.astype(bool)
    x1 = values[treated]
    x0 = values[~treated]
    return standardized_mean_difference(
        float(np.mean(x1)),
        float(np.mean(x0)),
        float(np.std(x1, ddof=1)),
        float(np.std(x0, ddof=1)),
    )


def balance_from_arm_stats(arm_stats: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Build a balance table from a two-row arm summary.

    ``arm_stats`` must contain ``treatment`` plus ``mean_<var>`` and ``std_<var>``
    for every variable.

    Args:
        arm_stats: One row per treatment arm.
        variables: Names that match the ``mean_`` / ``std_`` prefixes.

    Returns:
        One row per variable with means, standard deviations, and SMD.
    """
    treated = arm_stats.loc[arm_stats["treatment"] == 1].iloc[0]
    control = arm_stats.loc[arm_stats["treatment"] == 0].iloc[0]
    rows: list[dict[str, float | str]] = []
    for name in variables:
        mean_t = float(treated[f"mean_{name}"])
        mean_c = float(control[f"mean_{name}"])
        std_t = float(treated[f"std_{name}"])
        std_c = float(control[f"std_{name}"])
        smd = standardized_mean_difference(mean_t, mean_c, std_t, std_c)
        role = "pre_treatment" if name in FEATURE_COLUMNS else "post_treatment"
        rows.append(
            {
                "variable": name,
                "role": role,
                "mean_treatment": mean_t,
                "mean_control": mean_c,
                "std_treatment": std_t,
                "std_control": std_c,
                "smd": smd,
                "abs_smd": abs(smd),
                "imbalanced": abs(smd) >= SMD_BALANCE_THRESHOLD,
            }
        )
    return pd.DataFrame(rows)


def balance_verdict(balance: pd.DataFrame, threshold: float = SMD_BALANCE_THRESHOLD) -> dict[str, float | str | bool]:
    """Answer whether pre-treatment assignment looks balanced.

    Args:
        balance: Table produced by :func:`balance_from_arm_stats`.
        threshold: Absolute SMD that flags a covariate.

    Returns:
        Verdict, maximum absolute SMD, and the number of flagged covariates.
    """
    pre = balance.loc[balance["role"] == "pre_treatment"]
    max_abs = float(pre["abs_smd"].max())
    n_flagged = int(pre["imbalanced"].sum())
    balanced = bool(max_abs < threshold)
    if balanced:
        text = (
            "Yes. Pre-treatment covariates are balanced: the largest absolute "
            f"standardized difference is {max_abs:.4f}, below the {threshold:.2f} threshold. "
            "Treatment assignment is consistent with the randomized design, so the "
            "unadjusted difference in outcomes is a credible causal estimate."
        )
    else:
        flagged = ", ".join(pre.loc[pre["imbalanced"], "variable"].tolist())
        text = (
            "Pre-treatment balance is weaker than the usual rule of thumb. "
            f"{n_flagged} covariate(s) exceed |SMD| {threshold:.2f}: {flagged}. "
            "Regression adjustment and doubly robust estimation are the right sensitivity checks."
        )
    post = balance.loc[balance["role"] == "post_treatment"]
    if not post.empty:
        worst = post.sort_values("abs_smd", ascending=False).iloc[0]
        text += (
            f" {worst['variable']} is post-treatment and is imbalanced "
            f"(SMD {float(worst['smd']):.3f}); it is excluded from every model."
        )
    return {
        "balanced": balanced,
        "max_abs_smd_pretreatment": max_abs,
        "n_flagged_pretreatment": n_flagged,
        "threshold": threshold,
        "verdict": text,
    }


def missingness_table(quality: pd.DataFrame) -> pd.DataFrame:
    """Reshape a one-row data-quality query into a missingness table."""
    row = quality.iloc[0]
    n = float(row["n_rows"])
    records = []
    for column in quality.columns:
        if not column.startswith("missing_"):
            continue
        name = column.replace("missing_", "")
        count = int(row[column])
        records.append({"variable": name, "n_missing": count, "pct_missing": count / n if n else np.nan})
    return pd.DataFrame(records)


def extreme_value_flags(extremes: pd.DataFrame, ratio: float = 8.0) -> pd.DataFrame:
    """Flag features whose maximum is far beyond the 99th percentile.

    Args:
        extremes: Columns ``feature``, ``p01``, ``p99``, ``max_value``, ``min_value``.
        ratio: Flag when the upper tail exceeds this multiple of the p01–p99 span.

    Returns:
        The same table with a boolean ``extreme_upper_tail`` column.
    """
    frame = extremes.copy()
    span = (frame["p99"] - frame["p01"]).abs().clip(lower=1e-8)
    frame["extreme_upper_tail"] = frame["max_value"] > frame["p99"] + ratio * span
    frame["extreme_lower_tail"] = frame["min_value"] < frame["p01"] - ratio * span
    return frame
