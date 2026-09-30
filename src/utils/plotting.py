"""Matplotlib figures saved for the README and the dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#1F4E79"
TEAL = "#0F6E6B"
ORANGE = "#D4762C"
RED = "#A33B32"
GRAY = "#5C6B73"
LIGHT = "#D9E2EC"


def apply_style() -> None:
    """Set a consistent, print-friendly style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6EDF2",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save ``fig`` and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def love_plot(balance: pd.DataFrame, path: Path, threshold: float = 0.1) -> None:
    """Horizontal standardized-mean-difference plot."""
    apply_style()
    plot_df = balance.sort_values("smd").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    colors = [RED if abs(v) >= threshold or role != "pre_treatment" else NAVY for v, role in zip(plot_df["smd"], plot_df["role"])]
    ax.axvspan(-threshold, threshold, color=TEAL, alpha=0.08, label=f"|SMD| < {threshold:.2f}")
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.axvline(-threshold, color=TEAL, linewidth=1, linestyle="--")
    ax.axvline(threshold, color=TEAL, linewidth=1, linestyle="--")
    ax.scatter(plot_df["smd"], np.arange(len(plot_df)), c=colors, s=42, zorder=3)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["variable"])
    ax.set_xlabel("Standardized mean difference (treatment − control)")
    ax.set_title("Covariate balance")
    ax.legend(loc="lower right")
    save_figure(fig, path)


def outcome_rates_plot(summary: pd.DataFrame, path: Path) -> None:
    """Grouped bars of visit and conversion rates by arm."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    arms = ["control", "treatment"]
    for ax, outcome, title in zip(axes, ["visit_rate", "conversion_rate"], ["Visit rate", "Conversion rate"]):
        values = [float(summary.loc[summary["arm"] == arm, outcome].iloc[0]) for arm in arms]
        ax.bar(arms, values, color=[ORANGE, NAVY], width=0.72)
        ax.set_title(title)
        ax.set_ylabel("Rate")
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.4%}" if value < 0.02 else f"{value:.2%}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Outcomes by randomly assigned arm", y=1.02)
    save_figure(fig, path)


def propensity_plot(e: np.ndarray, t: np.ndarray, path: Path) -> None:
    """Overlap of cross-fit propensity scores."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(e[t == 0], bins=bins, density=True, alpha=0.65, color=ORANGE, label="Control")
    ax.hist(e[t == 1], bins=bins, density=True, alpha=0.65, color=NAVY, label="Treatment")
    ax.set_xlabel("Cross-fit propensity score P(treatment = 1 | X)")
    ax.set_ylabel("Density")
    ax.set_title("Propensity score overlap")
    ax.legend()
    save_figure(fig, path)


def weight_plot(weights: np.ndarray, path: Path) -> None:
    """Distribution of stabilized IPW weights."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    hi = float(np.quantile(weights, 0.995))
    ax.hist(weights[weights <= hi], bins=40, color=TEAL, alpha=0.9)
    ax.set_xlabel("Stabilized weight")
    ax.set_ylabel("Users")
    ax.set_title("Stabilized inverse-probability weights")
    save_figure(fig, path)


def ate_forest_plot(estimates: pd.DataFrame, path: Path, title: str) -> None:
    """Point estimates and 95% confidence intervals."""
    apply_style()
    plot_df = estimates.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = np.arange(len(plot_df))
    colors = [RED if not flag else NAVY for flag in plot_df["causal"]]
    ax.hlines(y, plot_df["ci_low"], plot_df["ci_high"], color=colors, linewidth=2)
    ax.scatter(plot_df["estimate"], y, c=colors, s=46, zorder=3)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["name"])
    ax.set_xlabel("Risk difference (treatment − control)")
    ax.set_title(title)
    save_figure(fig, path)


def histogram_plot(
    values: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    vline: float | None = None,
) -> None:
    """Single-series histogram with an optional reference line."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.hist(values, bins=40, color=NAVY, alpha=0.9)
    if vline is not None:
        ax.axvline(vline, color=ORANGE, linewidth=2, label=f"Observed {vline:.5f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    save_figure(fig, path)


def qini_plot(curves: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    """Qini curves for one or more scores, plus the random-targeting line."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    palette = [NAVY, TEAL, ORANGE]
    for color, (name, frame) in zip(palette, curves.items()):
        ax.plot(frame["fraction"], frame["qini"], label=name, color=color, linewidth=2)
    reference = next(iter(curves.values()))
    final = float(reference["qini"].iloc[-1])
    ax.plot([0, 1], [0, final], color=GRAY, linestyle="--", label="Random targeting")
    ax.set_xlabel("Fraction of users targeted (ranked by predicted uplift)")
    ax.set_ylabel("Qini (incremental conversions in the treated sample)")
    ax.set_title(title)
    ax.legend()
    save_figure(fig, path)


def decile_plot(deciles: pd.DataFrame, path: Path, title: str) -> None:
    """Empirical uplift by predicted-uplift decile, with Wald intervals."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = deciles["decile"].to_numpy()
    y = deciles["empirical_uplift"].to_numpy()
    yerr = 1.96 * deciles["uplift_se"].to_numpy()
    ax.bar(x, y, color=NAVY, alpha=0.85, width=0.72)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=ORANGE, capsize=3, linewidth=1.2)
    if "mean_predicted_cate" in deciles:
        ax.plot(x, deciles["mean_predicted_cate"], color=TEAL, marker="o", label="Mean predicted CATE")
    if "mean_true_cate" in deciles:
        ax.plot(x, deciles["mean_true_cate"], color=ORANGE, marker="o", label="Mean true CATE")
    if "mean_predicted_cate" in deciles or "mean_true_cate" in deciles:
        ax.legend()
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set_xlabel("Predicted uplift decile (1 = lowest)")
    ax.set_ylabel("Empirical risk difference")
    ax.set_title(title)
    save_figure(fig, path)


def policy_plot(curve: pd.DataFrame, path: Path, value: float, cost: float, chosen: float, title: str) -> None:
    """Profit and captured incremental outcomes against the targeting fraction."""
    apply_style()
    profit = curve["incremental_conversions"] * value - curve["users_targeted"] * cost
    fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
    ax1.plot(curve["target_fraction"], profit, color=NAVY, linewidth=2, label="Incremental profit")
    ax1.axhline(0, color=GRAY, linewidth=1)
    ax1.axvline(chosen, color=ORANGE, linestyle="--", linewidth=1.4, label=f"Validation choice {chosen:.0%}")
    ax1.set_xlabel("Fraction targeted")
    ax1.set_ylabel("Incremental profit on the test set (USD)")
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(
        curve["target_fraction"],
        curve["incremental_conversions"],
        color=TEAL,
        linewidth=1.6,
        alpha=0.85,
        label="Incremental outcomes",
    )
    ax2.set_ylabel("Empirical incremental outcomes")
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    save_figure(fig, path)


def subgroup_plot(table: pd.DataFrame, path: Path, title: str) -> None:
    """Quintile uplift for the features with the widest spread."""
    apply_style()
    features = list(dict.fromkeys(table["feature"]))
    fig, axes = plt.subplots(1, len(features), figsize=(3.1 * len(features), 4.2), sharey=True)
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features):
        part = table[table["feature"] == feature]
        ax.bar(part["bin"].astype(str), part["uplift"], color=NAVY, width=0.72)
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.set_title(feature)
        ax.set_xlabel("Quintile")
    axes[0].set_ylabel("Empirical risk difference")
    fig.suptitle(title, y=1.03)
    save_figure(fig, path)


def cate_hist(scores: np.ndarray, path: Path, title: str) -> None:
    """Distribution of out-of-sample predicted treatment effects."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.hist(scores, bins=40, color=TEAL, alpha=0.92)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.axvline(float(np.mean(scores)), color=ORANGE, linewidth=2, label=f"Mean {np.mean(scores):.5f}")
    ax.set_xlabel("Predicted individual treatment effect")
    ax.set_ylabel("Users in the test set")
    ax.set_title(title)
    ax.legend()
    save_figure(fig, path)
