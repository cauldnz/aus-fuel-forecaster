"""Central configuration. All paths and constants live here.

CLAUDE.md forbids hard-coded paths in pipeline modules — they must come from
this file (or be passed in via CLI arguments).
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------- Paths -----------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_INTERIM: Path = DATA_DIR / "interim"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_STATIC: Path = DATA_DIR / "static"

MODELS_DIR: Path = REPO_ROOT / "models"
RESULTS_DIR: Path = REPO_ROOT / "results"
SHAP_DIR: Path = RESULTS_DIR / "shap"

# ----------------------------- Span -----------------------------

# v1 historical span. FuelCheck monthly archives start 2016-09 and the
# project covers data up to the most recent complete month at run time.
SPAN_START: str = "2016-09-01"

# ----------------------------- Train / val / test folds -----------------------------
# Per spec.md §8.3.

TRAIN_END: str = "2022-12-31"
VAL_START: str = "2023-01-01"
VAL_END: str = "2023-12-31"
TEST_START: str = "2024-01-01"
TEST_NORMAL_END: str = "2025-12-31"
TEST_CRISIS_START: str = "2026-01-01"  # Reported separately as out-of-distribution.

# ----------------------------- Day-of-fortnight anchor -----------------------------
# Anchor for `cal_day_of_fortnight` per spec.md §7.3.
# 2016-07-04 is a Monday and predates the FuelCheck history window.
DOF_ANCHOR: str = "2016-07-04"

# ----------------------------- Fuels -----------------------------

FUELS_V1: tuple[str, ...] = ("U91", "DL")  # Unleaded 91, Diesel.

# ----------------------------- Network -----------------------------

USER_AGENT: str = "fuel-pred/0.1 (https://github.com/cauldnz/fuel-prediction)"

REQUEST_TIMEOUT: int = 30
RETRY_MAX_ATTEMPTS: int = 5
RETRY_BACKOFF_SECONDS: float = 2.0

# ----------------------------- Modeling -----------------------------

LGBM_PARAMS: dict[str, object] = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 2000,
    "early_stopping_rounds": 100,
    "verbose": -1,
    "random_state": 42,
}

# ----------------------------- Augmentor variables -----------------------------
# Keys are the aliases used in the augmentor call (and become column names with
# the `sa2_` prefix). Values are the GCP DataPack variable references or a
# marker indicating "derive in src/build/enrich_census.py".

AUGMENTOR_VARIABLES: dict[str, str] = {
    "median_age": "G02.Median_age_persons",
    "median_household_income_weekly": "G02.Median_tot_hhd_inc_weekly",
    "total_population": "G01.Tot_P_P",
    # The remaining variables are computed as ratios in enrich_census.py
    # because the augmentor exposes counts; we want percentages.
    "pct_drive_to_work": "DERIVED.from_G46",
    "motor_vehicles_per_dwelling": "DERIVED.from_G31",
    "pct_renters": "DERIVED.from_G33",
    "pct_employed_full_time": "DERIVED.from_G43",
    "pct_aged_65_plus": "DERIVED.from_G04",
    "pct_one_parent_family": "DERIVED.from_G25",
    # SEIFA is joined separately because the augmentor doesn't expose it
    # in the GCP DataPack; see src/build/enrich_census.py.
    "seifa_irsd_score": "EXTERNAL.seifa_2021_irsd",
}
