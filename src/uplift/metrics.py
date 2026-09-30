"""Out-of-sample uplift curves.

The Qini curve follows the Radcliffe construction: rank by predicted uplift,
then compare cumulative outcomes in the treated and control users inside that
ranking. The area between that curve and the random-targeting line is the
Qini coefficient. It is computed only on rows the model did not train on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ranking_order(score: np.ndarray, seed: int = 42) -> np.ndarray:
    """Descending score order with a seeded jitter to break ties."""
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0, 1e-10, size=len(score))
    return np.argsort(-(score.astype(np.float64) + jitter), kind="mergesort")


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray, seed: int = 42) -> pd.DataFrame:
    """Qini values along the predicted-uplift ranking.

    Args:
        y: Held-out outcomes.
        t: Held-out treatment.
        score: Predicted uplift. Higher means treat first.
        seed: Tie-break seed.

    Returns:
        Fraction targeted and Qini value after each user.
    """
    order = ranking_order(score, seed)
    y_sorted = y[order].astype(float)
    t_sorted = t[order].astype(float)
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1.0 - t_sorted)
    cum_yt = np.cumsum(y_sorted * t_sorted)
    cum_yc = np.cumsum(y_sorted * (1.0 - t_sorted))
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = cum_yt - cum_yc * cum_t / cum_c
    valid = (cum_t > 0) & (cum_c > 0)
    qini = np.where(valid, qini, np.nan)
    fraction = np.arange(1, len(y) + 1) / len(y)
    return pd.DataFrame({"fraction": fraction, "qini": qini})


def _with_origin(curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    qini = curve["qini"].to_numpy(dtype=float)
    # The first rows can be undefined until both arms have appeared.
    finite = np.where(np.isfinite(qini), qini, 0.0)
    fraction = np.concatenate([[0.0], curve["fraction"].to_numpy(dtype=float)])
    values = np.concatenate([[0.0], finite])
    return fraction, values


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray, seed: int = 42) -> float:
    """Area between the model Qini curve and the random-targeting line.

    Args:
        y: Held-out outcomes.
        t: Held-out treatment.
        score: Predicted uplift.
        seed: Tie-break seed.

    Returns:
        Qini coefficient. Positive means the ranking beats random targeting.
    """
    curve = qini_curve(y, t, score, seed)
    fraction, values = _with_origin(curve)
    area = float(np.trapezoid(values, fraction))
    final = float(values[-1])
    random_area = 0.5 * final
    return area - random_area


def downsample_qini(curve: pd.DataFrame, n_points: int = 250) -> pd.DataFrame:
    """Keep an evenly spaced subset of Qini points, including the endpoint."""
    idx = np.linspace(0, len(curve) - 1, n_points).astype(int)
    idx = np.unique(idx)
    frame = curve.iloc[idx].copy()
    frame["qini"] = frame["qini"].ffill().fillna(0.0)
    return frame.reset_index(drop=True)


def bootstrap_qini_coefficient(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    n_boot: int = 40,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile interval for the Qini coefficient, resampling held-out rows.

    The model is not refit. This interval captures evaluation noise, not
    training noise.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # Require both arms so a resample can still form a contrast.
        if np.unique(t[idx]).size < 2:
            stats[i] = np.nan
            continue
        stats[i] = qini_coefficient(y[idx], t[idx], score[idx], seed)
    clean = stats[np.isfinite(stats)]
    return {
        "mean": float(np.mean(clean)),
        "ci_low": float(np.quantile(clean, 0.025)),
        "ci_high": float(np.quantile(clean, 0.975)),
    }


def decile_calibration(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    n_bins: int = 10,
    truth: np.ndarray | None = None,
) -> pd.DataFrame:
    """Empirical risk difference inside predicted-uplift deciles.

    Args:
        y: Held-out outcomes.
        t: Held-out treatment.
        score: Predicted uplift.
        n_bins: Number of quantile bins.

    Returns:
        Decile number, mean prediction, empirical uplift, and its standard error.
    """
    bins = pd.qcut(score, q=n_bins, labels=False, duplicates="drop")
    rows = []
    for decile in sorted(pd.unique(bins)):
        mask = bins == decile
        yt = y[mask & (t == 1)]
        yc = y[mask & (t == 0)]
        if len(yt) == 0 or len(yc) == 0:
            continue
        p1 = float(yt.mean())
        p0 = float(yc.mean())
        uplift = p1 - p0
        se = float(np.sqrt(p1 * (1 - p1) / len(yt) + p0 * (1 - p0) / len(yc)))
        rows.append(
            {
                "decile": int(decile) + 1,
                "n": int(mask.sum()),
                "n_treatment": int(len(yt)),
                "n_control": int(len(yc)),
                "mean_predicted_cate": float(score[mask].mean()),
                "empirical_uplift": uplift,
                "uplift_se": se,
                "ci_low": uplift - 1.959963984540054 * se,
                "ci_high": uplift + 1.959963984540054 * se,
            }
        )
        if truth is not None:
            rows[-1]["mean_true_cate"] = float(truth[mask].mean())
    return pd.DataFrame(rows)


def decile_rank_correlation(deciles: pd.DataFrame) -> float:
    """Spearman correlation between decile index and empirical uplift."""
    if len(deciles) < 3:
        return float("nan")
    corr = spearmanr(deciles["decile"], deciles["empirical_uplift"]).statistic
    return float(corr)
