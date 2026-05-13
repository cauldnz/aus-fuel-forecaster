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
# Keys are the aliases passed to ``census_augment.Pipeline.create(variables=...)``
# and become DataFrame column names prefixed with ``sa2_`` (the augmentor's
# default ``output_prefix``). Values are augmentor variable references in the
# ``<NAMESPACE>.<field>`` form — see each dataset's spec markdown in the
# augmentor repo (`datasets/<id>.md`) for the canonical schema.
#
# Spec: spec.md §7.7. The order here mirrors the spec block order.

AUGMENTOR_VARIABLES: dict[str, str] = {
    # Census 2021 GCP — direct fields
    "median_age": "G02.Median_age_persons",
    "median_household_income_weekly": "G02.Median_tot_hhd_inc_weekly",
    "total_population": "G01.Tot_P_P",
    # Census 2021 PRESETs — six curated ratios with their right denominators
    # baked in; resolves the long-standing "what's the right denominator
    # per column" spike (augmentor #11, #18, #23 history in spec §7.7.1).
    "pct_drive_to_work": "PRESET.pct_drive_to_work",
    "motor_vehicles_per_dwelling": "PRESET.motor_vehicles_per_dwelling",
    "pct_renters": "PRESET.pct_renters",
    "pct_employed_full_time": "PRESET.pct_employed_full_time",
    "pct_aged_65_plus": "PRESET.pct_aged_65_plus",
    "pct_one_parent_family": "PRESET.pct_one_parent_family",
    # SEIFA 2021 — four indexes, score values (technical paper recommends
    # quantiles over scores for modelling but we keep the score for
    # finer-grained tree splits). State-relative deciles deferred until
    # we see whether the score scale alone gives the model enough signal.
    "seifa_irsd_score": "SEIFA.irsd_score",
    "seifa_irsad_score": "SEIFA.irsad_score",
    "seifa_ier_score": "SEIFA.ier_score",
    "seifa_ieo_score": "SEIFA.ieo_score",
    # ABS Estimated Resident Population — latest annual release (currently
    # 2024). The augmentor's ERP fetcher only exposes a single point-in-time
    # value (`population_total`) plus per-year history columns; the dataset
    # spec markdown promises age bands / density / median age but those
    # aren't wired up in v1.5 (verified empirically — see PR #46 description
    # for the upstream-issue pointer). Useful signal here is the
    # post-Census drift: ERP `population_total` (2024) vs `G01.Tot_P_P`
    # (2021) lets the model see growth corridors that Census alone misses.
    "erp_population_total": "ERP.population_total",
    # ABS Personal Income in Australia — latest financial-year release
    # (currently 2022-23). LEED-derived from ATO data, so different bias
    # profile to Census's self-report household income (`G02.Median_tot_hhd_inc_weekly`):
    # ABS_PIA captures the high-income tail without top-coding, but
    # excludes non-filers (low end). Both signals worth keeping. Note: the
    # dataset spec markdown promises `gini_coefficient` but the v1.5
    # fetcher only emits these 4 summary stats.
    "pia_median_total_income": "ABS_PIA.median_total_income",
    "pia_mean_total_income": "ABS_PIA.mean_total_income",
    "pia_income_earners_count": "ABS_PIA.income_earners_count",
    "pia_median_age_of_earners": "ABS_PIA.median_age_of_earners",
    # DSS Payment Demographic Data — latest quarter (currently 2025-Q3),
    # snapshot pinned. SA2-level recipient counts, not rates; the model
    # picks up per-station scaling via interaction with the §7.5 stn block.
    # Per-row temporal resolution deferred (spec §7.7.2). Selected from the
    # ~21 columns DSS publishes per quarter — the ones excluded (e.g.
    # ABSTUDY, special benefit, austudy, low-income card) have very small
    # recipient pops that suppress to null in most NSW SA2s.
    "dss_age_pension_recipients": "DSS.age_pension_recipients",
    "dss_jobseeker_payment_recipients": "DSS.jobseeker_payment_recipients",
    "dss_disability_support_pension_recipients": "DSS.disability_support_pension_recipients",
    "dss_parenting_payment_single_recipients": "DSS.parenting_payment_single_recipients",
    "dss_parenting_payment_partnered_recipients": "DSS.parenting_payment_partnered_recipients",
    "dss_carer_payment_recipients": "DSS.carer_payment_recipients",
    "dss_carer_allowance_recipients": "DSS.carer_allowance_recipients",
    "dss_youth_allowance_other_recipients": "DSS.youth_allowance_other_recipients",
    "dss_youth_allowance_student_and_apprentice_recipients": (
        "DSS.youth_allowance_student_and_apprentice_recipients"
    ),
    "dss_commonwealth_rent_assistance_recipients": "DSS.commonwealth_rent_assistance_recipients",
    "dss_commonwealth_seniors_health_card_recipients": (
        "DSS.commonwealth_seniors_health_card_recipients"
    ),
    "dss_family_tax_benefit_a_recipients": "DSS.family_tax_benefit_a_recipients",
    "dss_family_tax_benefit_b_recipients": "DSS.family_tax_benefit_b_recipients",
}
