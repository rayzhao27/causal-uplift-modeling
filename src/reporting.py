"""Narrative text built only from computed analysis outputs."""

from __future__ import annotations

import math
from typing import Any


def percent(value: float, digits: int = 3) -> str:
    """Format a rate as a percentage."""
    return f"{100 * value:.{digits}f}%"


def points(value: float, digits: int = 3) -> str:
    """Format a risk difference in percentage points."""
    return f"{100 * value:.{digits}f} percentage points"


def p_value(value: float) -> str:
    """Format a p-value, including underflow from a very large sample."""
    if value is None or not math.isfinite(value):
        return "undefined"
    if value == 0 or value < 1e-300:
        return "< 1e-300"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def usd(value: float) -> str:
    """Format a dollar amount."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _estimator_sentence(estimates: list[dict[str, Any]]) -> str:
    causal = [row for row in estimates if row.get("causal", True)]
    values = [float(row["estimate"]) for row in causal]
    spread = max(values) - min(values)
    names = ", ".join(f"{row['name']} {points(float(row['estimate']))}" for row in causal)
    sample = next(row for row in causal if row["name"] == "Sample difference in means")
    if spread <= 2 * float(sample["se"]):
        relation = (
            "These causal estimators agree to within about two standard errors of the "
            "sample difference in means. That is what a randomized experiment should produce: "
            "with a nearly constant propensity score, regression adjustment, inverse-probability "
            "weighting, and the doubly robust estimator all target the same risk difference."
        )
    else:
        relation = (
            "The estimators are farther apart than sampling error alone would suggest. "
            "In a clean randomized experiment that usually means a misspecified adjustment model, "
            "unstable weights, or a covariate that is not pre-treatment."
        )
    return f"On the modeling sample the estimates are {names}. The spread from lowest to highest is {points(spread)}. {relation}"


def build_recommendation(summary: dict[str, Any]) -> str:
    """Turn computed results into the decision the dashboard leads with.

    Args:
        summary: Analysis payload. Every quantity is read from this object.

    Returns:
        A recommendation grounded in the experiment and the held-out policy evaluation.
    """
    conversion = summary["experiment"]["conversion"]
    visit = summary["experiment"]["visit"]
    balance = summary["balance"]["verdict"]
    conv_policy = summary["policy"]["conversion"]
    visit_policy = summary["policy"]["visit"]
    conv_uplift = summary["uplift"]["conversion"]
    visit_uplift = summary["uplift"]["visit"]

    lines = [
        (
            f"The campaign's intent-to-treat effect on conversion is {points(conversion['absolute_lift'])} "
            f"({percent(conversion['rate_control'])} in control, {percent(conversion['rate_treatment'])} in treatment, "
            f"95% CI {points(conversion['ci_low'])} to {points(conversion['ci_high'])}, "
            f"p {p_value(conversion['z_pvalue'])}). "
            f"Relative to the control rate that is a {percent(conversion['relative_lift'], 1)} relative lift, "
            f"or {conversion['absolute_lift'] * 10000:.2f} incremental conversions per 10,000 treated users. "
            f"Achieved power at the observed effect is {conversion['achieved_power']:.3f}, and the "
            f"minimum detectable effect at 80% power is {points(conversion['mde_80_power'])}. "
            "With a sample this large, a tiny lift can be statistically significant; the operating "
            "question is whether the lift covers the cost of treatment."
        ),
        (
            f"The same assignment increases visits by {points(visit['absolute_lift'])} "
            f"({percent(visit['rate_control'])} vs {percent(visit['rate_treatment'])}, "
            f"95% CI {points(visit['ci_low'])} to {points(visit['ci_high'])}). "
            "Visit is an engagement outcome, not a substitute for conversion."
        ),
        balance,
        _estimator_sentence(summary["ate"]["conversion"]),
    ]

    def targeting_sentence(label: str, uplift: dict[str, Any], policy: dict[str, Any]) -> str:
        selected = uplift["selected_model"]
        qini = uplift["models"][selected]["test_qini"]
        low = uplift["models"][selected]["test_qini_ci_low"]
        high = uplift["models"][selected]["test_qini_ci_high"]
        fraction = policy["selected_fraction_validation"]
        test = policy["test_performance"]
        treat_all = policy["treat_all_test"]
        if not uplift["beats_random"]:
            return (
                f"Out-of-sample {label} ranking does not reliably beat random targeting. "
                f"The validation-selected model is {selected}, with a test Qini coefficient of {qini:.4f} "
                f"(95% interval {low:.4f} to {high:.4f}). "
                "Do not deploy an individual targeting score for this outcome; use the average effect and unit economics."
            )
        if fraction <= 0:
            return (
                f"The {selected} ranks {label} better than random on the held-out test "
                f"(Qini coefficient {qini:.4f}, 95% interval {low:.4f} to {high:.4f}), "
                f"but under the assumed economics (cost {usd(policy['cost'])} per treated user, "
                f"value {usd(policy['value'])} per incremental {label}) every targeting fraction "
                "loses money on the validation set. The recommendation is not to treat."
            )
        share = test.get("share_of_treat_all_incremental")
        share_text = "a share that is undefined because treating everyone is not incremental" if share is None else f"{100 * share:.1f}% of the incremental {label}s from treating everyone"
        return (
            f"For {label}, select users with the {selected}, which won on the validation Qini coefficient. "
            f"On the locked test set its Qini coefficient is {qini:.4f} (95% interval {low:.4f} to {high:.4f}). "
            f"The fraction chosen on validation is the top {fraction:.0%}. "
            f"On the test set that policy targets {int(test['users_targeted']):,} users, "
            f"produces {test['incremental_conversions']:.1f} incremental {label}s "
            f"({test['incremental_conversions_per_1000']:.2f} per 1,000 targeted), and captures {share_text}. "
            f"At the assumed economics its test-set profit is {usd(test['incremental_profit'])}, "
            f"compared with {usd(treat_all['incremental_profit'])} from treating everyone in the test set."
        )

    if "ground_truth" in summary:
        truth = summary["ground_truth"]["conversion"]
        lines.append(
            f"The data are simulated, so the conversion effect is known. "
            f"The average true effect is {points(truth['mean_true_cate'])}; "
            f"the experiment estimate differs from it by {points(truth['ate_error'])}. "
            f"On the held-out test set the selected conversion model has Spearman correlation "
            f"{truth['test_spearman']:.2f} with the true individual effect."
        )
    lines.append(targeting_sentence("conversion", conv_uplift, conv_policy))
    lines.append(targeting_sentence("visit", visit_uplift, visit_policy))

    conv_be = conv_policy["breakeven_value_per_outcome"]
    if conv_be is None:
        lines.append("The average conversion effect is not positive, so treating everyone has no break-even value.")
    else:
        lines.append(
            f"Break-even, ignoring heterogeneity: at a treatment cost of {usd(conv_policy['cost'])}, "
            f"a conversion must be worth at least {usd(conv_be)} for treating everyone to pay for itself. "
            f"The analysis assumes {usd(conv_policy['value'])}. "
            "That value is an assumption, not an estimate from the experiment."
        )
    chosen = conv_policy["selected_fraction_validation"]
    chosen_profit = conv_policy["test_performance"]["incremental_profit"]
    best_test_profit = conv_policy["exploratory_test_best_profit"]
    best_test_fraction = conv_policy["exploratory_test_best_fraction"]
    if conv_uplift["beats_random"] and chosen > 0 and chosen_profit > 0:
        lines.append(
            f"Decision: keep the campaign, but do not treat the full list. "
            f"Contact the top {chosen:.0%} of users by the conversion uplift score. "
            "That fraction was chosen on the validation set and is profitable on the locked test set."
        )
    elif conversion["absolute_lift"] > 0 and conv_policy["treat_all_test"]["incremental_profit"] > 0:
        lines.append(
            "Decision: the average conversion effect is positive and, on the held-out test set, "
            "treating everyone is profitable under the stated assumptions. "
            "Individual targeting is not reliable enough to replace that uniform policy."
        )
    elif conv_uplift["beats_random"] and chosen > 0 and chosen_profit <= 0:
        if best_test_profit <= 0:
            lines.append(
                f"Decision: do not launch this offer. The conversion lift is real, and the uplift score "
                f"ranks users better than chance, but the validation-chosen top {chosen:.0%} loses "
                f"{usd(abs(chosen_profit))} on the locked test set. The least-negative point on the "
                f"test-set profit curve is the top {best_test_fraction:.0%}, at {usd(best_test_profit)}. "
                "Treating nobody earns zero and beats every targeting fraction under the stated cost and value. "
                "Revisit the offer only if the contact cost falls or the margin per conversion rises enough "
                "to clear the break-even value above."
            )
        else:
            lines.append(
                f"Decision: do not deploy the validation-chosen top {chosen:.0%}. "
                f"That rule loses {usd(abs(chosen_profit))} on the locked test set. "
                f"A higher test-set profit appears at the top {best_test_fraction:.0%}, "
                "but that point was found by looking at the test curve, so it is not a confirmed policy."
            )
    else:
        lines.append(
            "Decision: do not scale a treat-everyone conversion campaign under the stated cost and value. "
            "Revisit the decision only if the margin per conversion is above the break-even value, "
            "or if a new experiment shows a stable uplift ranking."
        )
    return " ".join(lines)
