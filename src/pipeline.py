"""End-to-end analysis: experiment, causal estimates, uplift, and targeting."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.causal.ate import (
    aipw_ate,
    cross_fit_nuisance,
    difference_in_means,
    full_sample_itt,
    ipw_ate,
    regression_adjusted_ate,
    weight_diagnostics,
)
from src.causal.robustness import (
    bootstrap_difference_in_means,
    placebo_treatment,
    random_common_cause,
    subsample_stability,
)
from src.data.query import query_experiment, read_sql_file
from src.data.synthetic import materialize_experiment
from src.reporting import build_recommendation
from src.stats.ab_test import experiment_table, power_report, two_proportion_inference
from src.stats.balance import balance_from_arm_stats, balance_verdict, extreme_value_flags, missingness_table
from src.uplift.metrics import (
    bootstrap_qini_coefficient,
    decile_calibration,
    decile_rank_correlation,
    downsample_qini,
    qini_coefficient,
    qini_curve,
)
from src.uplift.models import RLearner, TLearner
from src.uplift.policy import choose_fraction, lookup_fraction, policy_curve, profit_of_row, scale_to_cohort
from src.utils.config import (
    COST_PER_TREATMENT,
    FEATURE_COLUMNS,
    FIGURES,
    OUTCOMES,
    POLICY_GRID,
    POLICY_SELECTION_GRID,
    EXPERIMENT_PARQUET,
    N_USERS,
    SEED,
    SQL_DIR,
    TABLES,
    TRAIN_FRACTION,
    VAL_FRACTION,
    VALUE_PER_OUTCOME,
    covariate_columns,
)
from src.utils.io import read_json, write_json
from src.utils.plotting import (
    ate_forest_plot,
    cate_hist,
    decile_plot,
    histogram_plot,
    love_plot,
    outcome_rates_plot,
    policy_plot,
    propensity_plot,
    qini_plot,
    subgroup_plot,
    weight_plot,
)

CHECKPOINT = TABLES / "checkpoint.json"


def log(message: str) -> None:
    """Print a single progress line."""
    print(message, flush=True)


def _deltas_for(base_rate: float) -> list[float]:
    if base_rate < 0.02:
        return [0.0002, 0.0005, 0.001, 0.002]
    return [0.002, 0.005, 0.01, 0.02]


def _as_int(value: object) -> int:
    return int(value)


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = covariate_columns()
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Covariate matrix contains non-finite values.")
    return values


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = 1 - TRAIN_FRACTION - VAL_FRACTION
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=SEED,
        stratify=frame["treatment"],
    )
    relative_val = VAL_FRACTION / (TRAIN_FRACTION + VAL_FRACTION)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=SEED,
        stratify=train_val["treatment"],
    )
    parts = [part.reset_index(drop=True) for part in (train, val, test)]
    ids = [set(part["row_id"].tolist()) for part in parts]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise RuntimeError("Train, validation, and test row ids overlap.")
    return parts[0], parts[1], parts[2]


def _segment_uplift(segments: pd.DataFrame, rate_column: str) -> pd.DataFrame:
    rows = []
    for (feature, bin_id), part in segments.groupby(["feature", "bin"], sort=True):
        treated = part.loc[part["treatment"] == 1, rate_column]
        control = part.loc[part["treatment"] == 0, rate_column]
        if treated.empty or control.empty:
            continue
        rows.append(
            {
                "feature": feature,
                "bin": int(bin_id),
                "uplift": float(treated.iloc[0] - control.iloc[0]),
                "rate_treatment": float(treated.iloc[0]),
                "rate_control": float(control.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _widest_features(uplift: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    spread = uplift.groupby("feature")["uplift"].agg(lambda s: float(s.max() - s.min())).sort_values(ascending=False)
    keep = list(spread.head(k).index)
    return uplift[uplift["feature"].isin(keep)].copy()


def _json_safe_frame(frame: pd.DataFrame) -> list[dict[str, object]]:
    return frame.replace({np.nan: None}).to_dict(orient="records")


def phase_experiment(n_users: int) -> dict[str, object]:
    """Generate the experiment and run the SQL diagnostics."""
    log("Phase 1 — generating the experiment and running SQL diagnostics")
    materialize_experiment(n=n_users, seed=SEED, force=True)
    overview = query_experiment(read_sql_file(SQL_DIR / "01_experiment_overview.sql"))
    by_arm = query_experiment(read_sql_file(SQL_DIR / "02_outcomes_by_arm.sql"))
    arm_stats = query_experiment(read_sql_file(SQL_DIR / "03_covariate_summary_by_arm.sql"))
    quality = query_experiment(read_sql_file(SQL_DIR / "04_data_quality.sql"))
    extremes = extreme_value_flags(query_experiment(read_sql_file(SQL_DIR / "06_feature_extremes.sql")))
    row = overview.iloc[0]
    n_rows = _as_int(row["n_rows"])
    n_t = _as_int(row["n_treatment"])
    n_c = _as_int(row["n_control"])
    if n_t + n_c != n_rows:
        raise RuntimeError("Treatment and control counts do not add up to the experiment size.")
    for outcome in OUTCOMES:
        counted = _as_int(row[f"{outcome}_treatment"]) / n_t
        reported = float(row[f"{outcome}_rate_treatment"])
        if abs(counted - reported) > 1e-10:
            raise RuntimeError(f"{outcome} treatment rate does not match count / n.")

    balance = balance_from_arm_stats(arm_stats, list(FEATURE_COLUMNS) + ["exposure"])
    verdict = balance_verdict(balance)
    missing = missingness_table(quality)
    log(
        f"  n={n_rows:,}  treatment={n_t:,} ({n_t / n_rows:.1%})  control={n_c:,}  "
        f"max |SMD|={verdict['max_abs_smd_pretreatment']:.4f}"
    )
    log(f"  {verdict['verdict']}")

    segments = query_experiment(read_sql_file(SQL_DIR / "05_segment_conversion_rates.sql"))
    conversion_segments = _segment_uplift(segments, "conversion_rate")
    visit_segments = _segment_uplift(segments, "visit_rate")

    TABLES.mkdir(parents=True, exist_ok=True)
    balance.to_csv(TABLES / "covariate_balance.csv", index=False)
    missing.to_csv(TABLES / "missingness.csv", index=False)
    extremes.to_csv(TABLES / "feature_extremes.csv", index=False)
    by_arm.to_csv(TABLES / "outcomes_by_arm.csv", index=False)
    segments.to_csv(TABLES / "segment_rates.csv", index=False)
    rates = by_arm.rename(columns={"treatment": "arm"})
    rates["arm"] = rates["arm"].map({0: "control", 1: "treatment"})
    outcome_rates_plot(rates, FIGURES / "outcome_rates_by_arm.png")
    love_plot(balance.loc[balance["role"] == "pre_treatment"], FIGURES / "love_plot.png")
    subgroup_plot(
        _widest_features(conversion_segments),
        FIGURES / "subgroup_uplift_conversion.png",
        "Largest pre-treatment quintile differences in conversion",
    )
    return {
        "overview": {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()},
        "quality": {key: (None if pd.isna(value) else value) for key, value in quality.iloc[0].to_dict().items()},
        "balance": verdict,
        "n_extreme_features": int((extremes["extreme_upper_tail"] | extremes["extreme_lower_tail"]).sum()),
        "n_users": n_rows,
    }


def phase_ab_test(overview: dict[str, object]) -> dict[str, dict[str, object]]:
    """Full-experiment intent-to-treat tests and power."""
    log("Phase 2 — intent-to-treat tests")
    n_t = _as_int(overview["n_treatment"])
    n_c = _as_int(overview["n_control"])
    results: dict[str, dict[str, object]] = {}
    tables = []
    for outcome in OUTCOMES:
        effect = two_proportion_inference(
            _as_int(overview[f"{outcome}_treatment"]),
            n_t,
            _as_int(overview[f"{outcome}_control"]),
            n_c,
            outcome=outcome,
        )
        power = power_report(effect, _deltas_for(effect.rate_control))
        results[outcome] = {**effect.as_dict(), **{k: v for k, v in power.items() if k != "outcome"}}
        tables.append(experiment_table(effect, power))
        log(
            f"  {outcome}: lift {effect.absolute_lift:.6f} "
            f"CI [{effect.ci_low:.6f}, {effect.ci_high:.6f}] "
            f"z={effect.z_statistic:.2f} p={effect.z_pvalue:.3g} "
            f"power={power['achieved_power']:.3f} mde={power['mde_80_power']:.6f}"
        )
    pd.concat(tables, ignore_index=True).to_csv(TABLES / "experiment_results.csv", index=False)
    return results


def _estimate_ates(frame: pd.DataFrame, overview: dict[str, object]) -> tuple[dict[str, list[dict[str, object]]], dict[str, float], np.ndarray, np.ndarray]:
    x = _matrix(frame)
    treatment = frame["treatment"].to_numpy(dtype=float)
    n_t = _as_int(overview["n_treatment"])
    n_c = _as_int(overview["n_control"])
    stored: dict[str, list[dict[str, object]]] = {}
    diagnostics: dict[str, float] = {}
    propensity_for_plot = None
    for outcome in OUTCOMES:
        y = frame[outcome].to_numpy(dtype=float)
        log(f"Phase 3 — causal estimates for {outcome}")
        propensity, mu0, mu1 = cross_fit_nuisance(x, y, treatment)
        if propensity_for_plot is None:
            propensity_for_plot = propensity
            diagnostics = weight_diagnostics(treatment, propensity)
            diagnostics["propensity_auc"] = float(roc_auc_score(treatment.astype(int), propensity))
            if np.any(~np.isfinite(propensity)) or propensity.min() <= 0 or propensity.max() >= 1:
                raise RuntimeError("Propensity scores are outside (0, 1).")
            log(
                f"  propensity AUC={diagnostics['propensity_auc']:.4f}  "
                f"weight p99={diagnostics['weight_p99']:.3f}  "
                f"ESS={diagnostics['effective_sample_fraction']:.1%}"
            )
        estimates = [
            full_sample_itt(_as_int(overview[f"{outcome}_treatment"]), n_t, _as_int(overview[f"{outcome}_control"]), n_c),
            difference_in_means(y, treatment),
            regression_adjusted_ate(x, y, treatment),
            ipw_ate(y, treatment, propensity),
            aipw_ate(y, treatment, propensity, mu0, mu1),
        ]
        if outcome == "conversion":
            leaked = np.column_stack(
                [x, frame["visit"].to_numpy(dtype=float), frame["exposure"].to_numpy(dtype=float)]
            )
            estimates.append(
                regression_adjusted_ate(
                    leaked,
                    y,
                    treatment,
                    name="Not causal: controls for visit and exposure",
                    causal=False,
                    notes="Post-treatment controls. Shown only to illustrate the bias, not as an effect estimate.",
                )
            )
        stored[outcome] = [item.as_dict() for item in estimates]
        causal_values = [item.estimate for item in estimates if item.causal]
        log(
            "  "
            + " | ".join(f"{item.name.split('(')[0].strip()} {item.estimate:.6f}" for item in estimates if item.causal)
            + f" | spread {max(causal_values) - min(causal_values):.6f}"
        )
        ate_forest_plot(
            pd.DataFrame(stored[outcome]),
            FIGURES / f"ate_comparison_{outcome}.png",
            f"Average treatment effect on {outcome}",
        )
    if propensity_for_plot is None:
        raise RuntimeError("Propensity scores were not estimated.")
    propensity_plot(propensity_for_plot, treatment, FIGURES / "propensity_scores.png")
    weight_plot(
        np.where(
            treatment == 1,
            float(treatment.mean()) / np.clip(propensity_for_plot, 0.01, 0.99),
            (1 - float(treatment.mean())) / (1 - np.clip(propensity_for_plot, 0.01, 0.99)),
        ),
        FIGURES / "ipw_weights.png",
    )
    rows = []
    for outcome, estimates in stored.items():
        for estimate in estimates:
            rows.append({"outcome": outcome, **estimate})
    pd.DataFrame(rows).to_csv(TABLES / "ate_comparison.csv", index=False)
    return stored, diagnostics, propensity_for_plot, treatment


def _robustness(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    log("Phase 3b — robustness checks")
    x = _matrix(frame)
    treatment = frame["treatment"].to_numpy(dtype=float)
    output: dict[str, dict[str, object]] = {}
    for outcome in OUTCOMES:
        y = frame[outcome].to_numpy(dtype=float)
        placebo = placebo_treatment(y, treatment, n_perm=200, seed=SEED)
        boot = bootstrap_difference_in_means(y, treatment, n_boot=200, seed=SEED)
        noise = random_common_cause(x, y, treatment, seed=SEED)
        subsets = subsample_stability(y, treatment, n_reps=30, fraction=0.25, seed=SEED)
        subsets.to_csv(TABLES / f"subsample_stability_{outcome}.csv", index=False)
        histogram_plot(
            placebo["placebo_distribution"],
            FIGURES / f"placebo_ate_{outcome}.png",
            f"Placebo treatment effects on {outcome}",
            "Permuted risk difference",
            vline=float(placebo["observed_ate"]),
        )
        histogram_plot(
            boot["distribution"],
            FIGURES / f"bootstrap_ate_{outcome}.png",
            f"Bootstrap distribution of the {outcome} risk difference",
            "Bootstrap risk difference",
            vline=float(difference_in_means(y, treatment).estimate),
        )
        output[outcome] = {
            "placebo_mean": placebo["placebo_mean"],
            "placebo_std": placebo["placebo_std"],
            "placebo_p_value": placebo["placebo_p_value"],
            "bootstrap_se": boot["boot_se"],
            "bootstrap_ci_low": boot["ci_low"],
            "bootstrap_ci_high": boot["ci_high"],
            "noise_covariate_baseline": noise["baseline_ate"],
            "noise_covariate_augmented": noise["augmented_ate"],
            "noise_covariate_absolute_change": noise["absolute_change"],
            "subsample_min": float(subsets["estimate"].min()),
            "subsample_max": float(subsets["estimate"].max()),
            "subsample_std": float(subsets["estimate"].std(ddof=1)),
        }
        log(
            f"  {outcome}: placebo p={placebo['placebo_p_value']:.4f}  "
            f"noise |change|={noise['absolute_change']:.6f}  "
            f"subsample range [{output[outcome]['subsample_min']:.6f}, {output[outcome]['subsample_max']:.6f}]"
        )
    return output


def _fit_outcome(
    outcome: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, object]:
    log(f"Phase 4 — uplift models for {outcome}")
    packs = {}
    for name, part in (("train", train), ("validation", val), ("test", test)):
        packs[name] = (
            _matrix(part),
            part[outcome].to_numpy(dtype=float),
            part["treatment"].to_numpy(dtype=float),
        )
    x_train, y_train, t_train = packs["train"]
    x_val, y_val, t_val = packs["validation"]
    x_test, y_test, t_test = packs["test"]
    if set(train["row_id"]).intersection(test["row_id"]):
        raise RuntimeError("Uplift training rows leaked into the test set.")
    fitted: dict[str, dict[str, object]] = {}
    qini_frames = []
    for name, model in (("T-learner", TLearner(seed=SEED)), ("R-learner", RLearner(seed=SEED))):
        started = time.perf_counter()
        model.fit(x_train, y_train, t_train)
        score_val = model.predict(x_val)
        score_test = model.predict(x_test)
        val_q = qini_coefficient(y_val, t_val, score_val, seed=SEED)
        test_q = qini_coefficient(y_test, t_test, score_test, seed=SEED)
        curve = downsample_qini(qini_curve(y_test, t_test, score_test, seed=SEED))
        curve.insert(0, "model", name)
        qini_frames.append(curve)
        fitted[name] = {
            "val_qini": val_q,
            "test_qini": test_q,
            "score_val": score_val,
            "score_test": score_test,
        }
        log(f"  {name}: val Qini {val_q:.2f}  test Qini {test_q:.2f}  ({time.perf_counter() - started:.0f}s)")
    selected = max(fitted, key=lambda name: fitted[name]["val_qini"])
    selected_test = fitted[selected]["score_test"]
    qini_ci = bootstrap_qini_coefficient(y_test, t_test, selected_test, n_boot=40, seed=SEED)
    beats_random = bool(qini_ci["ci_low"] > 0)
    truth_column = f"true_cate_{outcome}"
    truth = test[truth_column].to_numpy(dtype=float) if truth_column in test.columns else None
    deciles = decile_calibration(y_test, t_test, selected_test, truth=truth)
    spearman = decile_rank_correlation(deciles)
    deciles.to_csv(TABLES / f"deciles_{outcome}.csv", index=False)
    counts, edges = np.histogram(selected_test, bins=40)
    pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts}).to_csv(
        TABLES / f"cate_hist_{outcome}.csv", index=False
    )
    qini_curves = {name: frame.drop(columns="model") for name, frame in zip(("T-learner", "R-learner"), qini_frames)}
    ground_truth: dict[str, float] | None = None
    if truth is not None:
        from scipy.stats import spearmanr

        oracle = downsample_qini(qini_curve(y_test, t_test, truth, seed=SEED))
        oracle.insert(0, "model", "Oracle")
        qini_frames.append(oracle)
        qini_curves["Oracle"] = oracle.drop(columns="model")
        spearman_truth = float(spearmanr(selected_test, truth).statistic)
        rmse = float(np.sqrt(np.mean((selected_test - truth) ** 2)))
        ground_truth = {"spearman": spearman_truth, "rmse": rmse, "mean_true_cate": float(np.mean(truth))}
        log(f"  true-CATE Spearman {spearman_truth:.3f}  RMSE {rmse:.4f}")
    pd.concat(qini_frames, ignore_index=True).to_csv(TABLES / f"qini_{outcome}.csv", index=False)
    qini_plot(qini_curves, FIGURES / f"qini_{outcome}.png", f"Test-set Qini curve for {outcome}")
    decile_plot(deciles, FIGURES / f"decile_uplift_{outcome}.png", f"Test-set uplift by predicted {outcome} decile")
    cate_hist(selected_test, FIGURES / f"cate_distribution_{outcome}.png", f"Test-set predicted effects on {outcome}")
    model_metrics = {
        name: {
            "val_qini": float(payload["val_qini"]),
            "test_qini": float(payload["test_qini"]),
            "test_qini_ci_low": float(qini_ci["ci_low"]) if name == selected else None,
            "test_qini_ci_high": float(qini_ci["ci_high"]) if name == selected else None,
        }
        for name, payload in fitted.items()
    }
    log(f"  selected {selected}; test Qini interval [{qini_ci['ci_low']:.2f}, {qini_ci['ci_high']:.2f}]; beats random={beats_random}")
    return {
        "selected_model": selected,
        "beats_random": beats_random,
        "decile_spearman": spearman,
        "models": model_metrics,
        "y_val": y_val,
        "t_val": t_val,
        "y_test": y_test,
        "t_test": t_test,
        "score_val": fitted[selected]["score_val"],
        "score_test": selected_test,
        "split_sizes": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "ground_truth": ground_truth,
    }


def _policy_for_outcome(outcome: str, fitted: dict[str, object]) -> dict[str, object]:
    value = VALUE_PER_OUTCOME[outcome]
    cost = COST_PER_TREATMENT
    val_curve = policy_curve(
        fitted["y_val"], fitted["t_val"], fitted["score_val"], POLICY_GRID, seed=SEED
    )
    test_curve = policy_curve(
        fitted["y_test"], fitted["t_test"], fitted["score_test"], POLICY_GRID, seed=SEED
    )
    full = test_curve.loc[np.isclose(test_curve["target_fraction"], 1.0)].iloc[0]
    ate = float(fitted["y_test"][fitted["t_test"] == 1].mean() - fitted["y_test"][fitted["t_test"] == 0].mean())
    if abs(float(full["uplift_rate"]) - ate) > 1e-8:
        raise RuntimeError(f"{outcome} policy at 100% does not match the test-set risk difference.")
    chosen = choose_fraction(val_curve, value=value, cost=cost, grid=POLICY_SELECTION_GRID)
    test_row = lookup_fraction(test_curve, chosen)
    treat_all = lookup_fraction(test_curve, 1.0)
    test_perf = {
        **{key: (None if pd.isna(val) else val) for key, val in test_row.to_dict().items()},
        "incremental_profit": 0.0 if chosen <= 0 else profit_of_row(test_row, value, cost),
        "campaign_cost": 0.0 if chosen <= 0 else float(test_row["users_targeted"]) * cost,
        "incremental_value": 0.0 if chosen <= 0 else float(test_row["incremental_conversions"]) * value,
    }
    treat_all_perf = {
        **{key: (None if pd.isna(val) else val) for key, val in treat_all.to_dict().items()},
        "incremental_profit": profit_of_row(treat_all, value, cost),
        "campaign_cost": float(treat_all["users_targeted"]) * cost,
        "incremental_value": float(treat_all["incremental_conversions"]) * value,
    }
    test_profits = test_curve.apply(lambda row: profit_of_row(row, value, cost), axis=1)
    best_idx = int(test_profits.idxmax())
    projection = scale_to_cohort(test_row, cohort_n=100_000, value=value, cost=cost)
    full_ate = None  # filled by caller if needed
    sensitivity = []
    for cost_option in (0.50, 1.25, 2.50):
        for value_option in (VALUE_PER_OUTCOME[outcome] / 2, VALUE_PER_OUTCOME[outcome], VALUE_PER_OUTCOME[outcome] * 2):
            fraction = choose_fraction(val_curve, value=value_option, cost=cost_option, grid=POLICY_SELECTION_GRID)
            evaluated = lookup_fraction(test_curve, fraction)
            profit = 0.0 if fraction <= 0 else profit_of_row(evaluated, value_option, cost_option)
            sensitivity.append(
                {
                    "cost": cost_option,
                    "value": value_option,
                    "selected_fraction_validation": fraction,
                    "test_incremental_conversions": float(evaluated["incremental_conversions"]),
                    "test_profit": profit,
                }
            )
    test_curve.to_csv(TABLES / f"policy_curve_test_{outcome}.csv", index=False)
    val_curve.to_csv(TABLES / f"policy_curve_validation_{outcome}.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(TABLES / f"policy_sensitivity_{outcome}.csv", index=False)
    policy_plot(
        test_curve,
        FIGURES / f"policy_profit_{outcome}.png",
        value=value,
        cost=cost,
        chosen=chosen,
        title=f"Test-set targeting value for {outcome}",
    )
    log(
        f"  {outcome}: validation chooses top {chosen:.0%}; "
        f"test incremental {test_perf['incremental_conversions']:.1f}; "
        f"profit {test_perf['incremental_profit']:.2f}"
    )
    return {
        "cost": cost,
        "value": value,
        "selected_fraction_validation": chosen,
        "exploratory_test_best_fraction": float(test_curve.loc[best_idx, "target_fraction"]),
        "exploratory_test_best_profit": float(test_profits.loc[best_idx]),
        "test_performance": test_perf,
        "treat_all_test": treat_all_perf,
        "cohort_100k": projection,
        "breakeven_value_per_outcome": full_ate,
        "sensitivity": sensitivity,
        "_full_test_ate": float(full["uplift_rate"]),
    }


def _attach_breakeven(policy: dict[str, dict[str, object]], experiment: dict[str, dict[str, object]]) -> None:
    for outcome, payload in policy.items():
        ate = float(experiment[outcome]["absolute_lift"])
        payload["breakeven_value_per_outcome"] = None if ate <= 0 else COST_PER_TREATMENT / ate
        payload.pop("_full_test_ate", None)


def run(from_phase: int = 1, n_users: int = N_USERS) -> dict[str, object]:
    """Run the analysis and write tables, figures, and summary.json.

    Args:
        from_phase: First phase to execute. Later phases reload earlier artifacts.
        n_users: Users in the simulated experiment.

    Returns:
        The summary payload written to disk.
    """
    started = time.perf_counter()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    if from_phase <= 1:
        experiment_meta = phase_experiment(n_users)
        experiment = phase_ab_test(experiment_meta["overview"])
        write_json(
            CHECKPOINT,
            {"experiment_meta": experiment_meta, "experiment": experiment, "n_users": n_users},
        )
    else:
        checkpoint = read_json(CHECKPOINT)
        experiment_meta = checkpoint["experiment_meta"]
        experiment = checkpoint["experiment"]
        if int(checkpoint["n_users"]) != int(n_users):
            raise RuntimeError("Checkpoint size does not match --n-users. Rerun from phase 1.")

    sample = pd.read_parquet(EXPERIMENT_PARQUET)
    sample["treatment"] = sample["treatment"].astype(int)
    for column in list(OUTCOMES) + ["exposure", "visit"]:
        if column in sample.columns:
            sample[column] = sample[column].astype(float)

    if from_phase <= 3:
        ate, diagnostics, _, _ = _estimate_ates(sample, experiment_meta["overview"])
        robustness = _robustness(sample)
        checkpoint = read_json(CHECKPOINT)
        checkpoint.update({"ate": ate, "ipw": diagnostics, "robustness": robustness})
        write_json(CHECKPOINT, checkpoint)
    else:
        checkpoint = read_json(CHECKPOINT)
        ate = checkpoint["ate"]
        diagnostics = checkpoint["ipw"]
        robustness = checkpoint["robustness"]

    train, val, test = _split(sample)
    log(
        f"  split train={len(train):,} validation={len(val):,} test={len(test):,} "
        f"(treatment share {sample['treatment'].mean():.1%})"
    )
    fitted = {outcome: _fit_outcome(outcome, train, val, test) for outcome in OUTCOMES}
    log("Phase 5 — targeting policy")
    policy = {outcome: _policy_for_outcome(outcome, payload) for outcome, payload in fitted.items()}
    _attach_breakeven(policy, experiment)

    uplift_summary = {
        outcome: {
            "selected_model": payload["selected_model"],
            "beats_random": payload["beats_random"],
            "decile_spearman": payload["decile_spearman"],
            "models": payload["models"],
            "split_sizes": payload["split_sizes"],
            "ground_truth": payload["ground_truth"],
        }
        for outcome, payload in fitted.items()
    }
    ground_truth = {}
    for outcome in OUTCOMES:
        truth_mean = float(sample[f"true_cate_{outcome}"].mean())
        full = next(row for row in ate[outcome] if row["name"].startswith("Full"))
        row_level = next(row for row in ate[outcome] if row["name"].startswith("Sample"))
        if abs(float(full["estimate"]) - float(row_level["estimate"])) > 1e-8:
            raise RuntimeError(f"{outcome} SQL aggregate does not match the row-level risk difference.")
        ate_error = abs(float(full["estimate"]) - truth_mean)
        log(f"  {outcome} ATE minus mean true CATE: {float(full['estimate']) - truth_mean:.5f}")
        if ate_error > 0.02:
            raise RuntimeError(f"{outcome} ATE is {ate_error:.4f} away from the known mean effect.")
        ground_truth[outcome] = {
            "mean_true_cate": truth_mean,
            "ate": float(full["estimate"]),
            "ate_error": ate_error,
            "test_spearman": fitted[outcome]["ground_truth"]["spearman"] if fitted[outcome]["ground_truth"] else None,
            "test_rmse": fitted[outcome]["ground_truth"]["rmse"] if fitted[outcome]["ground_truth"] else None,
        }
    summary: dict[str, object] = {
        "dataset": {
            "name": "Synthetic randomized promotion experiment",
            "n_rows": _as_int(experiment_meta["overview"]["n_rows"]),
            "n_treatment": _as_int(experiment_meta["overview"]["n_treatment"]),
            "n_control": _as_int(experiment_meta["overview"]["n_control"]),
            "treatment_share": float(experiment_meta["overview"]["treatment_share"]),
            "sample_n": int(len(sample)),
            "features": list(FEATURE_COLUMNS),
            "excluded_from_models": ["exposure", "visit", "conversion", "true_cate_conversion", "true_cate_visit"],
            "exposure_rate_treatment": float(experiment_meta["overview"]["exposure_rate_treatment"]),
            "exposure_rate_control": float(experiment_meta["overview"]["exposure_rate_control"]),
        },
        "quality": experiment_meta["quality"],
        "n_extreme_features": experiment_meta["n_extreme_features"],
        "balance": experiment_meta["balance"],
        "experiment": experiment,
        "ate": ate,
        "ipw": diagnostics,
        "robustness": robustness,
        "uplift": uplift_summary,
        "policy": policy,
        "ground_truth": ground_truth,
    }
    summary["recommendation"] = build_recommendation(summary)
    write_json(TABLES / "summary.json", summary)
    log(f"Done in {(time.perf_counter() - started) / 60:.1f} min")
    log(summary["recommendation"])
    return summary


def main() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the campaign uplift analysis.")
    parser.add_argument("--from-phase", type=int, default=1, choices=(1, 3, 4))
    parser.add_argument("--n-users", type=int, default=N_USERS)
    args = parser.parse_args()
    run(from_phase=args.from_phase, n_users=args.n_users)


if __name__ == "__main__":
    main()
