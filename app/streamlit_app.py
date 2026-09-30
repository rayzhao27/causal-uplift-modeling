"""Decision dashboard for the promotion campaign experiment.

The app reads artifacts written by ``python run_analysis.py``. It does not
refit models. Profit under edited cost and value is recomputed from the
held-out uplift rates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
SUMMARY_PATH = TABLES / "summary.json"

NAVY = "#1F4E79"
TEAL = "#0F6E6B"
ORANGE = "#D4762C"
GRAY = "#5C6B73"


@st.cache_data
def load_summary() -> dict:
    """Load the analysis summary written by the pipeline."""
    return json.loads(SUMMARY_PATH.read_text())


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    """Load a saved CSV artifact."""
    return pd.read_csv(TABLES / name)


def _pct(value: float, digits: int = 3) -> str:
    return f"{100 * value:.{digits}f}%"


def _points(value: float) -> str:
    return f"{100 * value:.3f} pp"


def _p_value(value: float) -> str:
    if value is None:
        return "n/a"
    if value == 0 or value < 1e-300:
        return "< 1e-300"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _row_for_fraction(curve: pd.DataFrame, fraction: float) -> pd.Series:
    return curve.iloc[(curve["target_fraction"] - fraction).abs().argmin()]


def _profit_frame(curve: pd.DataFrame, value: float, cost: float) -> pd.DataFrame:
    frame = curve.dropna(subset=["uplift_rate"]).copy()
    frame["incremental_profit"] = frame["incremental_conversions"] * value - frame["users_targeted"] * cost
    frame["campaign_cost"] = frame["users_targeted"] * cost
    frame["incremental_value"] = frame["incremental_conversions"] * value
    return frame


def _line(x, y, name: str, color: str, dash: str | None = None) -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name, mode="lines", line={"color": color, "width": 2, "dash": dash})


def page_targeting(summary: dict) -> None:
    """Interactive targeting rule."""
    st.subheader("Who should receive the campaign?")
    st.caption(
        "Users are ranked by a model fit on the training split. "
        "The validation split chooses the model and the fraction. "
        "Every number below is the empirical treatment-control contrast on the locked test set."
    )
    outcome = st.selectbox("Outcome to optimize", ["conversion", "visit"], format_func=str.capitalize)
    policy = summary["policy"][outcome]
    uplift = summary["uplift"][outcome]
    left, right = st.columns(2)
    cost = left.number_input("Cost per treated user (USD)", min_value=0.0, value=float(policy["cost"]), step=0.01, format="%.2f")
    value = right.number_input(
        f"Value per incremental {outcome} (USD)",
        min_value=0.0,
        value=float(policy["value"]),
        step=1.0,
        format="%.2f",
    )
    curve = _profit_frame(load_table(f"policy_curve_test_{outcome}.csv"), value, cost)
    recommended = float(policy["selected_fraction_validation"])
    default_pct = int(round(recommended * 100)) if recommended > 0 else 20
    target_pct = st.slider("Target the top K% by predicted uplift", min_value=1, max_value=100, value=default_pct, step=1)
    row = _row_for_fraction(curve, target_pct / 100)
    users = int(row["users_targeted"])
    incremental = float(row["incremental_conversions"])
    per_thousand = float(row["incremental_conversions_per_1000"])
    share = row["share_of_treat_all_incremental"]
    profit = float(row["incremental_profit"])
    campaign_cost = float(row["campaign_cost"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Users targeted (test set)", f"{users:,}")
    m2.metric("Incremental outcomes", f"{incremental:,.1f}")
    m3.metric("Per 1,000 targeted", f"{per_thousand:.2f}")
    m4.metric("Share of treat-all incremental", "n/a" if pd.isna(share) else f"{100 * float(share):.1f}%")
    p1, p2, p3 = st.columns(3)
    p1.metric("Campaign cost", _money(campaign_cost))
    p2.metric("Incremental value", _money(float(row["incremental_value"])))
    p3.metric("Incremental profit", _money(profit))
    cohort_users = int(round(target_pct / 100 * 100_000))
    cohort_incremental = float(row["uplift_rate"]) * cohort_users
    st.caption(
        f"Projected to a future cohort of 100,000 similar users: treat {cohort_users:,}, "
        f"for about {cohort_incremental:,.1f} incremental {outcome}s and "
        f"{_money(cohort_incremental * value - cohort_users * cost)} incremental profit. "
        "This scales the test-set rate; it is not a second experiment."
    )

    profit_fig = go.Figure()
    profit_fig.add_trace(_line(curve["target_fraction"], curve["incremental_profit"], "Incremental profit", NAVY))
    profit_fig.add_vline(x=target_pct / 100, line_dash="dot", line_color=ORANGE)
    if recommended > 0:
        profit_fig.add_vline(x=recommended, line_dash="dash", line_color=TEAL)
    profit_fig.update_layout(
        title="Test-set profit by fraction targeted",
        xaxis_title="Fraction targeted",
        yaxis_title="Incremental profit (USD)",
        height=420,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"orientation": "h"},
    )
    st.plotly_chart(profit_fig, width="stretch")
    best = curve.loc[curve["incremental_profit"].idxmax()]
    best_profit = float(best["incremental_profit"])
    if best_profit <= 0:
        chart_note = (
            f"Every fraction on this chart loses money (the smallest loss is { _money(best_profit) } "
            f"at the top {float(best['target_fraction']):.0%}). Treating nobody earns zero and is the better action."
        )
    else:
        chart_note = (
            f"With the costs currently entered, the highest test-set profit is "
            f"{_money(best_profit)} at the top {float(best['target_fraction']):.0%}. "
            "That chart maximum was not used to choose the model."
        )
    st.caption(
        f"Validation-selected fraction for the pre-specified economics: {recommended:.0%}. {chart_note}"
    )
    selected = uplift["selected_model"]
    interval = uplift["models"][selected]
    beats = "The test-set interval is above zero, so the ranking beats random targeting." if uplift["beats_random"] else "The test-set interval includes zero or is negative, so the ranking does not reliably beat random targeting."
    st.info(
        f"Model selected on validation: {selected}. "
        f"Test Qini coefficient {interval['test_qini']:.2f} "
        f"(95% interval {interval['test_qini_ci_low']:.2f} to {interval['test_qini_ci_high']:.2f}). {beats}"
    )


def page_experiment(summary: dict) -> None:
    """Full-experiment intent-to-treat results."""
    data = summary["dataset"]
    st.subheader("Experiment overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Users", f"{data['n_rows']:,}")
    c2.metric("Treatment", f"{data['n_treatment']:,}")
    c3.metric("Control", f"{data['n_control']:,}")
    st.caption(
        f"Treatment share {data['treatment_share']:.1%}. "
        "Assignment is randomized, so the unadjusted outcome difference is the causal benchmark. "
        f"Exposure, which is not the assignment, is {data['exposure_rate_treatment']:.1%} in treatment "
        f"and {data['exposure_rate_control']:.1%} in control. Exposure is excluded from every model."
    )
    for outcome in ("conversion", "visit"):
        effect = summary["experiment"][outcome]
        st.markdown(f"**{outcome.capitalize()}**")
        a, b, c, d = st.columns(4)
        a.metric("Treatment rate", _pct(effect["rate_treatment"]))
        b.metric("Control rate", _pct(effect["rate_control"]))
        c.metric("Absolute lift", _points(effect["absolute_lift"]))
        d.metric("Relative lift", _pct(effect["relative_lift"], 1))
        st.write(
            f"95% CI {_points(effect['ci_low'])} to {_points(effect['ci_high'])}. "
            f"z = {effect['z_statistic']:.2f}, p {_p_value(effect['z_pvalue'])}, "
            f"chi-square = {effect['chi2_statistic']:.1f}. "
            f"Achieved power {effect['achieved_power']:.3f}. "
            f"Minimum detectable effect at 80% power: {_points(effect['mde_80_power'])}."
        )
    st.image(str(ROOT / "outputs" / "figures" / "outcome_rates_by_arm.png"))
    st.subheader("Was assignment balanced?")
    st.write(summary["balance"]["verdict"])
    st.image(str(ROOT / "outputs" / "figures" / "love_plot.png"))
    balance = load_table("covariate_balance.csv")
    st.dataframe(balance, width="stretch", hide_index=True)


def page_causal(summary: dict) -> None:
    """ATE comparison."""
    st.subheader("Causal effect")
    st.write(
        "The full-experiment difference in means is the primary benchmark. "
        "Regression adjustment, IPW, and AIPW use the same 120,000 users, "
        "with cross-fit propensity and outcome models. "
        "Agreement is expected because treatment was randomized. "
        "The red specification controls for visit and exposure and is not a causal estimate."
    )
    ipw = summary["ipw"]
    st.caption(
        f"Cross-fit propensity AUC {ipw['propensity_auc']:.3f}. "
        f"Stabilized weight 99th percentile {ipw['weight_p99']:.2f}. "
        f"Effective sample size {ipw['effective_sample_fraction']:.1%} of the experiment. "
        f"Share of scores clipped: {ipw['share_clipped']:.2%}."
    )
    for outcome in ("conversion", "visit"):
        rows = summary["ate"][outcome]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[row["estimate"] for row in rows],
                y=[row["name"] for row in rows],
                mode="markers",
                marker={"color": [NAVY if row["causal"] else "#A33B32" for row in rows], "size": 10},
                error_x={
                    "type": "data",
                    "array": [row["ci_high"] - row["estimate"] for row in rows],
                    "arrayminus": [row["estimate"] - row["ci_low"] for row in rows],
                    "color": GRAY,
                },
                showlegend=False,
            )
        )
        fig.add_vline(x=0, line_color=GRAY)
        fig.update_layout(
            title=f"Risk difference for {outcome}",
            xaxis_title="Treatment − control",
            height=380,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
        )
        st.plotly_chart(fig, width="stretch")
    robust = summary["robustness"]["conversion"]
    st.subheader("Robustness, conversion")
    st.write(
        f"Permuting treatment puts the observed effect in the tail of the placebo distribution "
        f"(p { _p_value(robust['placebo_p_value']) }). "
        f"Adding an irrelevant normal covariate changes the regression-adjusted estimate by "
        f"{robust['noise_covariate_absolute_change']:.6f}. "
        f"The bootstrap 95% interval is {_points(robust['bootstrap_ci_low'])} to {_points(robust['bootstrap_ci_high'])}. "
        f"On 25% subsamples the unadjusted estimate stays between {robust['subsample_min']:.6f} and {robust['subsample_max']:.6f}."
    )


def page_uplift(summary: dict) -> None:
    """Qini, deciles, and the predicted-effect distribution."""
    st.subheader("Uplift on held-out users")
    outcome = st.selectbox("Outcome", ["conversion", "visit"], format_func=str.capitalize, key="uplift_outcome")
    qini = load_table(f"qini_{outcome}.csv")
    fig = go.Figure()
    colors = {"T-learner": NAVY, "R-learner": TEAL}
    for name, part in qini.groupby("model"):
        fig.add_trace(_line(part["fraction"], part["qini"], name, colors.get(name, ORANGE)))
    final = float(qini["qini"].iloc[-1])
    fig.add_trace(_line([0, 1], [0, final], "Random targeting", GRAY, dash="dash"))
    fig.update_layout(
        title=f"Qini curve, test set, {outcome}",
        xaxis_title="Fraction targeted",
        yaxis_title="Qini",
        height=440,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"orientation": "h"},
    )
    st.plotly_chart(fig, width="stretch")
    deciles = load_table(f"deciles_{outcome}.csv")
    decile_fig = go.Figure()
    decile_fig.add_trace(
        go.Bar(
            x=deciles["decile"],
            y=deciles["empirical_uplift"],
            name="Empirical uplift",
            marker_color=NAVY,
            error_y={"type": "data", "array": 1.96 * deciles["uplift_se"], "color": ORANGE},
        )
    )
    decile_fig.add_trace(_line(deciles["decile"], deciles["mean_predicted_cate"], "Mean predicted CATE", TEAL))
    if "mean_true_cate" in deciles.columns:
        decile_fig.add_trace(_line(deciles["decile"], deciles["mean_true_cate"], "Mean true CATE", ORANGE))
    decile_fig.update_layout(
        title="Empirical uplift by predicted-effect decile",
        xaxis_title="Decile (1 = lowest predicted uplift)",
        yaxis_title="Risk difference",
        height=420,
        legend={"orientation": "h"},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    st.plotly_chart(decile_fig, width="stretch")
    hist = load_table(f"cate_hist_{outcome}.csv")
    hist_fig = go.Figure(go.Bar(x=(hist["bin_left"] + hist["bin_right"]) / 2, y=hist["count"], marker_color=TEAL))
    hist_fig.update_layout(
        title="Distribution of predicted treatment effects on the test set",
        xaxis_title="Predicted individual treatment effect",
        yaxis_title="Users",
        height=380,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    st.plotly_chart(hist_fig, width="stretch")
    uplift = summary["uplift"][outcome]
    truth = uplift.get("ground_truth") or {}
    truth_text = ""
    if truth.get("spearman") is not None:
        truth_text = (
            f" Spearman correlation with the known individual effect is {truth['spearman']:.2f} "
            f"(RMSE {truth['rmse']:.4f})."
        )
    st.write(
        f"Decile rank correlation between predicted bin and empirical uplift: {uplift['decile_spearman']:.2f}. "
        "A rising pattern means the score concentrates incremental outcomes. "
        "Flat bars mean the ranking is not identifying heterogeneous effects."
        + truth_text
    )


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="Campaign uplift targeting", layout="wide")
    st.title("Campaign targeting from a randomized experiment")
    st.caption("Randomized promotion experiment · intent-to-treat effects and out-of-sample uplift")
    if not SUMMARY_PATH.exists():
        st.error("No analysis outputs found. From the project root, run: python run_analysis.py")
        return
    summary = load_summary()
    st.write(summary["recommendation"])
    targeting, experiment, causal, uplift = st.tabs(["Targeting", "Experiment", "Causal effect", "Uplift"])
    with targeting:
        page_targeting(summary)
    with experiment:
        page_experiment(summary)
    with causal:
        page_causal(summary)
    with uplift:
        page_uplift(summary)


if __name__ == "__main__":
    main()
