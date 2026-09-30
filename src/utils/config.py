"""Project paths, seeds, and pre-specified analysis choices.

Economic assumptions are inputs, not quantities estimated from the experiment.
They were fixed before inspecting treatment effects.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
SQL_DIR = ROOT / "sql"

SEED = 42
N_USERS = 120_000
EXPERIMENT_PARQUET = DATA_PROCESSED / "experiment.parquet"

FEATURE_COLUMNS: tuple[str, ...] = (
    "age",
    "tenure_months",
    "prior_purchases",
    "prior_spend",
    "recency_days",
    "sessions_30d",
    "email_opens_30d",
    "pages_viewed_30d",
    "avg_order_value",
    "discount_sensitivity",
    "is_loyalty_member",
    "is_weekend_shopper",
    "device_mobile",
    "device_desktop",
)
TREATMENT_COL = "treatment"
OUTCOMES: tuple[str, ...] = ("conversion", "visit")
# exposure is generated from the assignment after randomization. visit and
# conversion are outcomes. True effects exist only because the data are simulated
# and are never passed to a model.
POST_TREATMENT_COLUMNS: tuple[str, ...] = (
    "exposure",
    "visit",
    "conversion",
    "true_cate_conversion",
    "true_cate_visit",
)

TRAIN_FRACTION = 0.55
VAL_FRACTION = 0.15
# The remainder is the locked test set.
N_FOLDS = 5
# Propensity scores outside this range are clipped for IPW stability.
PROPENSITY_CLIP = (0.01, 0.99)
# Linear models clip feature tails. Trees see the raw features.
WINSOR_QUANTILE = 0.001
SMD_BALANCE_THRESHOLD = 0.10

# Pre-specified unit economics (USD) for a promotional offer, not estimated
# from the outcomes. Cost is the fully loaded cost of treating one user.
COST_PER_TREATMENT = 1.25
VALUE_PER_OUTCOME = {
    "conversion": 40.0,
    "visit": 2.0,
}

POLICY_GRID = tuple(round(i / 100, 2) for i in range(1, 101))
POLICY_SELECTION_GRID = tuple(round(i / 100, 2) for i in range(5, 101, 5))


def covariate_columns() -> list[str]:
    """Pre-treatment covariates used in every causal and uplift model.

    Returns:
        Feature names. Post-treatment fields are intentionally absent.
    """
    forbidden = set(POST_TREATMENT_COLUMNS) | {TREATMENT_COL, "row_id"}
    columns = list(FEATURE_COLUMNS)
    overlap = forbidden.intersection(columns)
    if overlap:
        raise RuntimeError(f"Covariate list contains forbidden columns: {sorted(overlap)}")
    return columns
