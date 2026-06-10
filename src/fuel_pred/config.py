"""Central configuration. All paths and constants live here.

CLAUDE.md forbids hard-coded paths in pipeline modules — they must come from
this file (or be passed in via CLI arguments).

Secrets (API keys, OAuth tokens) live in a gitignored `.env` file at the
repo root. They're loaded on import via `python-dotenv`. See `.env.example`
for the template and the keys the project understands.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env from the repo root if present. find_dotenv would also search
# parent dirs but we want the project-local file specifically.
_REPO_ROOT_FOR_ENV = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT_FOR_ENV / ".env", override=False)

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

# Hand-curated static CSV of NSW major event venues. Seeded with the
# 10-venue pilot list from docs/research/2026-05_major_events_features.md
# to enable the spec §13.6 Phase 0 EDA gate without an API dependency.
# Owned by maintainer + agent (append-only, same policy as brand_aliases.csv).
STATIC_MAJOR_VENUES: Path = DATA_STATIC / "major_venues.csv"

# Per-station nearest-venue features (spec §13.6 Phase 1). Produced by
# `python -m fuel_pred.spatial.venues`. Optional input to make_features —
# when absent, the venue feature columns ship as nulls (consistent with
# the pattern used for aip_tgp / cash_rate / asx200 in §7.4).
INTERIM_STATIONS_VENUES: Path = DATA_INTERIM / "stations_venues.parquet"

# NOAA GFS/GEFS weather pipeline (spec §13.7 v2.0). Both paths are
# populated by Session 1/2/3 code:
#   - INTERIM_STATION_GRID_MAPPING: per-station 4-neighbour bilinear
#     weights against each of the three GFS/GEFS grid resolutions
#     (gfs 0.25°, gefs05 0.5°, gefs1 1°). One-shot computation via
#     ``python -m fuel_pred.spatial.gfs_grid``.
#   - RAW_WEATHER_GFS_DIR: per-(date, horizon) NSW-box grid parquets
#     named ``<YYYY-MM-DD>_h<N>.parquet``. Populated by
#     ``tools/parallel_gfs_fetch.py`` (Session 2).
INTERIM_STATION_GRID_MAPPING: Path = DATA_INTERIM / "station_grid_mapping.parquet"
RAW_WEATHER_GFS_DIR: Path = DATA_RAW / "weather_gfs"

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

# Historical Forecast API coverage start for Australia (empirically probed,
# preflight 2026-05). Below this date, ERA5 archive is used as a fallback in
# fetch.weather. See docs/research/2026-05_weather_leakage_preflight.md.
WEATHER_FORECAST_COVERAGE_START: str = "2017-01-01"

# Open-Meteo API key — loaded from .env (or environment). When set, the key
# is appended to every Open-Meteo request and raises the free-tier rate
# limits ~10x. None (no key) is fine but limits the parallel fetch
# throughput. Free registration at https://open-meteo.com/en/pricing.
OPENMETEO_API_KEY: str | None = os.environ.get("OPENMETEO_API_KEY") or None

# Weather data source selection (spec §13.7 v2.0).
#   "gfs"      — strict-free path: NOAA GFS/GEFS via anonymous AWS S3
#                byte-range subsetting. No API key, no quota, no 429s.
#                Default for users without an Open-Meteo paid plan.
#                Uses src/fuel_pred/fetch/gfs.py and the multi-horizon
#                grid-cell parquets under RAW_WEATHER_GFS_DIR.
#   "openmeteo"— optional paid-tier path: Open-Meteo Historical Forecast API.
#                Strongly recommended with OPENMETEO_API_KEY set, otherwise
#                free-tier rate limits make full-roster fetches infeasible
#                (empirically: 4,587 NSW stations cannot complete in one day
#                without a key; see docs/research/2026-05_weather_leakage_fix_resume_plan.md).
#                Uses src/fuel_pred/fetch/weather.py and per-station parquets.
#   "auto"     — pick "openmeteo" if OPENMETEO_API_KEY is set, else "gfs".
#                Maintains backward-compat for users on the paid plan while
#                defaulting new contributors to the strict-free path.
WEATHER_SOURCE: Literal["gfs", "openmeteo", "auto"] = os.environ.get(  # type: ignore[assignment]
    "WEATHER_SOURCE", "auto"
)


def resolve_weather_source() -> Literal["gfs", "openmeteo"]:
    """Resolve "auto" to the actual source based on key availability.

    Wrapped in a function (not just a module-level expression) so tests
    can monkeypatch WEATHER_SOURCE / OPENMETEO_API_KEY without import-time
    side-effects, and so the resolution is re-evaluated if env vars change
    at runtime (e.g., when the user adds a key without restarting).
    """
    src = os.environ.get("WEATHER_SOURCE", WEATHER_SOURCE)
    if src == "auto":
        key = os.environ.get("OPENMETEO_API_KEY") or OPENMETEO_API_KEY
        return "openmeteo" if key else "gfs"
    if src not in {"gfs", "openmeteo"}:
        raise ValueError(
            f"Invalid WEATHER_SOURCE={src!r}. Must be 'gfs', 'openmeteo', or 'auto'."
        )
    return src  # type: ignore[return-value]


# ----------------------------- Modeling -----------------------------

LGBM_PARAMS: dict[str, object] = {
    "objective": "regression_l1",
    "metric": "mae",
    # v3.0 tuned defaults from Optuna TPE sweep (Phase 3 #4, 200 trials,
    # 6-fold k-fold objective). Validated across 6 seeds: mean improvement
    # 0.170 c/L (WEAK WIN, |mean|/stdev = 1.29) vs the original v1/v2
    # defaults. See spec §8.2 + results/v3_phase3_hyperopt_validation.md.
    # Pattern: smaller more-regularized trees with more features per split
    # and no row bagging — model was over-fitting at the v1/v2 defaults.
    "learning_rate": 0.028,       # v3.0 (was 0.05)
    "num_leaves": 31,             # v3.0 (was 63)
    "min_data_in_leaf": 544,      # v3.0 (was 200)
    "feature_fraction": 0.85,     # v3.0 (was 0.8)
    "bagging_fraction": 0.69,     # v3.0 (was 0.8)
    "bagging_freq": 0,            # v3.0 (was 5) — no row bagging
    "lambda_l1": 0.059,           # v3.0 (was 0)
    "lambda_l2": 0.0,             # unchanged
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
# v2.0+ (spec §7.7.2) splits the augmentor surface into two passes:
#
# - CROSS_SECTIONAL: variables that don't temporal cleanly — frozen at
#   latest release per spec §7.7.5. Enriched onto stations.parquet via
#   build.enrich_census. Three reasons a variable belongs here:
#     1. GCP-routed (direct G##.* or GCP-internal PRESETs): upstream
#        augmentor #91 Stage 2 (proper per-release DataPacks routing) is
#        still on backlog; temporal mode raises a loud ValueError today.
#     2. ERP age/sex (population_65_plus, median_age): the source ABS
#        3235.0 cube only ships these for the latest publication year;
#        historical releases return null per the augmentor #92 docs.
#     3. Cross-dataset PRESETs that depend on ERP age/sex denominators
#        (pct_age_pension_recipients etc.): inherit (2) above.
#
# - TEMPORAL: variables that resolve per-row to the contemporaneous
#   release. Enriched onto data/interim/panel_sa2_temporal.parquet via
#   build.enrich_panel_temporal, then merged on (station_id, date) at
#   feature build time. Three families today:
#     - SEIFA 2016 + 2021 (~50/50 split across train fold).
#     - ERP population_total (annual back-projection via augmentor #92 fix,
#       2017+).
#     - DSS welfare (quarterly back to 2022-Q4 — gives val+test fold
#       per-quarter variation against a constant-2022-Q4 train fold).
#
# Spec: spec.md §7.7 — block schema; §7.7.2 — temporal split; §7.7.5 —
# v2.0 static-surface bump.

AUGMENTOR_VARIABLES_CROSS_SECTIONAL: dict[str, str] = {
    # Census 2021 GCP — direct fields
    "median_age": "G02.Median_age_persons",
    "median_household_income_weekly": "G02.Median_tot_hhd_inc_weekly",
    "total_population": "G01.Tot_P_P",
    # Census 2021 PRESETs — six curated ratios with their right denominators
    # baked in; resolves the long-standing "what's the right denominator
    # per column" spike (augmentor #11, #18, #23 history in spec §7.7.1).
    # GCP-internal so routes through the GCP DataPacks — must stay
    # cross-sectional (upstream #91 Stage 2 pending).
    "pct_drive_to_work": "PRESET.pct_drive_to_work",
    "motor_vehicles_per_dwelling": "PRESET.motor_vehicles_per_dwelling",
    "pct_renters": "PRESET.pct_renters",
    "pct_employed_full_time": "PRESET.pct_employed_full_time",
    "pct_aged_65_plus": "PRESET.pct_aged_65_plus",
    "pct_one_parent_family": "PRESET.pct_one_parent_family",
    # ABS ERP age/sex — latest release only. The 3235.0 cube doesn't ship
    # these for historical releases (augmentor #92 docs); they would be
    # null in temporal mode, so we keep them on the cross-sectional pass.
    "erp_population_65_plus": "ERP.population_65_plus",
    "erp_median_age": "ERP.median_age",
    # ABS Personal Income in Australia — latest financial-year release
    # (currently 2022-23). LEED-derived from ATO data, so different bias
    # profile to Census's self-report household income (`G02.Median_tot_hhd_inc_weekly`):
    # ABS_PIA captures the high-income tail without top-coding, but
    # excludes non-filers (low end). Both signals worth keeping. The dataset
    # spec markdown promises `gini_coefficient` + 4 income-by-source medians
    # but the v1.5 fetcher only parses Table 1.4 (the summary sheet); see
    # upstream issue #65. ABS_PIA also has a sparse temporal release
    # cadence and isn't worth the split here.
    "pia_median_total_income": "ABS_PIA.median_total_income",
    "pia_mean_total_income": "ABS_PIA.mean_total_income",
    "pia_income_earners_count": "ABS_PIA.income_earners_count",
    "pia_median_age_of_earners": "ABS_PIA.median_age_of_earners",
    # Cross-dataset PRESETs new in augmentor v2.0 (PR #86). Each pulls a
    # DSS numerator and an ERP age/sex denominator. They MUST stay
    # cross-sectional because of the ERP age/sex limitation (above) —
    # temporal mode would null them for non-latest-release rows.
    "pct_age_pension_recipients": "PRESET.pct_age_pension_recipients",
    "pct_jobseeker_recipients": "PRESET.pct_jobseeker_recipients",
    "welfare_density_index": "PRESET.welfare_density_index",
    # DSS Payment Demographic Data — latest quarter snapshot only. Per-row
    # temporal resolution attempted in PR B but blocked by an upstream
    # parser issue on the 2022-Q4 file (see AUGMENTOR_VARIABLES_TEMPORAL
    # comment for the cross-ref). Selected from the ~21 columns DSS
    # publishes per quarter — the ones excluded (ABSTUDY, special benefit,
    # austudy, low-income card) have very small recipient pops that
    # suppress to null in most NSW SA2s.
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

AUGMENTOR_VARIABLES_TEMPORAL: dict[str, str] = {
    # SEIFA — four indexes, score values. Augmentor v2.0 ships both 2016
    # (Edition 2) and 2021 (Edition 3) snapshots; temporal mode resolves
    # per-row to the contemporaneous release. Our 2016-09 → 2026-04 panel
    # gets a ~50/50 split (2017-2020 rows → 2016, 2021+ rows → 2021).
    "seifa_irsd_score": "SEIFA.irsd_score",
    "seifa_irsad_score": "SEIFA.irsad_score",
    "seifa_ier_score": "SEIFA.ier_score",
    "seifa_ieo_score": "SEIFA.ieo_score",
    # ABS ERP `population_total` — augmentor #92 closed via column
    # projection (PR #95). Per-row resolution back to release year 2017
    # against `population_history_<year>` cube data on Edition 3 (ABS
    # re-aggregates back-data via internal concordance). Pre-2017 rows
    # resolve to release 2017 (no earlier ERP release registered).
    "erp_population_total": "ERP.population_total",
    # NOTE: DSS variables are NOT in the temporal pass today.
    #
    # The augmentor's DSS XLSX parser fails on the 2022-Q4 release
    # (`RuntimeError: No SA2 data rows in dss-2022-Q4.xlsx`) — older
    # quarterly files have a different sheet/header layout that the
    # current parser doesn't handle. Filed upstream as
    # cauldnz/abs-census-augmentor#99 (see
    # tools/upstream_issue_v2_dss_2022q4_parser.md).
    #
    # Until that lands, DSS stays in AUGMENTOR_VARIABLES_CROSS_SECTIONAL
    # (latest quarter only), unchanged from PR A behaviour. Adding DSS
    # here later is a one-line move once the parser issue is resolved.
}

# Back-compat alias. Pre-PR-B code (and any old test) reads
# AUGMENTOR_VARIABLES as the full union. Keep the union exported so
# external scripts that probed the full surface keep working; modules
# that own one or the other pass should import the specific dict.
AUGMENTOR_VARIABLES: dict[str, str] = {
    **AUGMENTOR_VARIABLES_CROSS_SECTIONAL,
    **AUGMENTOR_VARIABLES_TEMPORAL,
}
