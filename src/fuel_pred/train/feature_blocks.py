"""Feature column groups for Models A and B (spec §7 + §8.4).

The make_features module emits ~80 columns into ``data/processed/features.parquet``.
Some are features for the model; some are identifiers / raw fields / today's
prices that would leak the target. This module defines:

- ``BLOCK_COLUMNS``: explicit column lists per spec §7 block — kept in lockstep
  with the names emitted by ``build/make_features.py``. We deliberately use
  explicit lists rather than prefix-globbing so a renamed column trips an
  assertion in tests instead of silently disappearing from the model.

- ``MODEL_A_BLOCKS`` / ``MODEL_B_BLOCKS``: per spec §8.4. The only difference
  is the ``sa2`` block.

- ``CATEGORICAL_COLUMNS``: passed to LightGBM via ``categorical_feature=`` so
  it builds set-membership splits instead of treating string labels as
  arbitrary integers.

- ``EXCLUDE_FROM_FEATURES``: columns that exist in features.parquet but
  must never reach the model (identifiers, raw text, target, today's price).

Spec: spec.md §7, §8.4.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# ---- Block definitions -----------------------------------------------------

# §7.1 Lag block — per (station_id, fuel_code) lag/rolling features for U91,
# plus cross-fuel features that join Diesel onto U91 rows.
LAG_COLUMNS: tuple[str, ...] = (
    "lag_price_1",
    "lag_price_2",
    "lag_price_3",
    "lag_price_7",
    "lag_price_14",
    "lag_price_28",
    "roll_price_mean_7",
    "roll_price_mean_14",
    "roll_price_mean_28",
    "roll_price_std_7",
    "roll_price_std_14",
    "days_since_last_price_change",
    "price_minus_28d_min",
    "price_minus_28d_max",
    "xfuel_dl_price_lag_0",
    "xfuel_dl_price_lag_1",
    "xfuel_u91_minus_dl_lag_1",
    "xfuel_dl_roll_mean_7",
)

# §7.2 Upstream block — Brent / AUD/USD / TGP commodity context.
UPSTREAM_COLUMNS: tuple[str, ...] = (
    "upstream_brent_lag_0",
    "upstream_brent_lag_1",
    "upstream_brent_lag_3",
    "upstream_brent_lag_7",
    "upstream_brent_lag_14",
    "upstream_audusd_lag_0",
    "upstream_audusd_lag_1",
    "upstream_audusd_lag_3",
    "upstream_audusd_lag_7",
    "upstream_brent_aud_lag_0",
    "upstream_brent_aud_lag_7",
    "upstream_brent_aud_lag_14",
    "upstream_brent_change_7d",
    "upstream_brent_change_14d",
    "upstream_audusd_change_7d",
    # AIP TGP — fail-soft Tier-2 source. Columns always present in the schema;
    # null when TGP data isn't available for that date.
    "upstream_tgp_sydney_lag_0",
    "upstream_tgp_sydney_lag_3",
    "upstream_tgp_sydney_lag_7",
    "upstream_tgp_minus_brent_aud_lag_7",
)

# §7.3 Calendar block.
CALENDAR_COLUMNS: tuple[str, ...] = (
    "cal_day_of_week",
    "cal_day_of_month",
    "cal_month",
    "cal_week_of_year",
    "cal_year",
    "cal_day_of_fortnight",
    "cal_is_public_holiday",
    "cal_days_to_next_public_holiday",
    "cal_days_since_last_public_holiday",
    "cal_is_school_holiday_nsw",
    "cal_is_first_business_day_after_break",
)

# §7.4 Demand context block — traffic + Tier-2 macro indicators.
CTX_COLUMNS: tuple[str, ...] = (
    "ctx_traffic_top1_distance_km",
    "ctx_traffic_top2_distance_km",
    "ctx_traffic_top3_distance_km",
    "ctx_traffic_top1_lag_1",
    "ctx_traffic_top1_lag_7",
    "ctx_traffic_top2_lag_1",
    "ctx_traffic_top2_lag_7",
    "ctx_traffic_top3_lag_1",
    "ctx_traffic_top3_lag_7",
    "ctx_traffic_5km_radius_count",
    "ctx_inflation_expectations_lag_7",
    "ctx_asx200_lag_1",
    "ctx_cash_rate",
)

# §7.5 Static station block — broadcast per station_id across the time index.
STN_COLUMNS: tuple[str, ...] = (
    "stn_brand_raw",
    "stn_brand_canonical",
    "stn_brand_is_major",
    "stn_is_franchisee",
    "stn_competitors_within_2km",
    "stn_competitors_within_5km",
    "stn_distance_to_sydney_terminal_km",
    "stn_is_metro",
)

# §7.6 Weather block — daily Open-Meteo aggregates per station coords.
WX_COLUMNS: tuple[str, ...] = (
    "wx_temp_max_c",
    "wx_temp_min_c",
    "wx_precipitation_mm",
    "wx_wind_speed_max_kmh",
    "wx_weather_code",
)

# §7.7 SA2 demographic block — the augmentor block; the ONLY difference
# between Models A and B.
#
# Note on PR #43 (reverted by PR #44): we briefly trimmed this block to
# drop 4 features whose linear correlation with Model A columns was
# |r| ≥ 0.5. That intuition was wrong. Linear correlation isn't a
# tree-model redundancy measure: LightGBM's `feature_fraction=0.8`
# randomly drops 20% of columns per tree specifically so it can extract
# independent signal from correlated inputs. Two features correlated in
# the population (e.g. `sa2_pct_renters` and `stn_competitors_within_2km`
# are both downstream of urban density) doesn't stop the model from
# using each one's residual signal in different splits. Restored.
#
# The right next move was adding NEW SA2 variables (more breadth, more
# orthogonal axes), which is what this PR does: 18 additional columns
# spanning 3 new datasets the augmentor v1.5 surface registered (DSS
# welfare, ERP demographics, ABS_PIA income inequality) plus three
# additional SEIFA scores. The block is kept in spec block-order.
SA2_COLUMNS: tuple[str, ...] = (
    # Census 2021 GCP — direct + PRESET ratios
    "sa2_total_population",
    "sa2_median_age",
    "sa2_median_household_income_weekly",
    "sa2_pct_drive_to_work",
    "sa2_motor_vehicles_per_dwelling",
    "sa2_pct_renters",
    "sa2_pct_employed_full_time",
    "sa2_pct_aged_65_plus",
    "sa2_pct_one_parent_family",
    # SEIFA 2021 — four indexes
    "sa2_seifa_irsd_score",
    "sa2_seifa_irsad_score",
    "sa2_seifa_ier_score",
    "sa2_seifa_ieo_score",
    # ABS Estimated Resident Population (annual; pinned to latest)
    "sa2_erp_population_density_per_km2",
    "sa2_erp_population_0_14",
    "sa2_erp_population_15_64",
    "sa2_erp_population_65_plus",
    "sa2_erp_median_age",
    # ABS Personal Income in Australia (annual; pinned to latest)
    "sa2_pia_gini_coefficient",
    # DSS Payment Demographic Data (quarterly; pinned to latest snapshot
    # for v1 — temporal per-row resolution deferred, see spec §7.7.2)
    "sa2_dss_age_pension_recipients",
    "sa2_dss_jobseeker_payment_recipients",
    "sa2_dss_disability_support_pension_recipients",
    "sa2_dss_parenting_payment_single_recipients",
    "sa2_dss_parenting_payment_partnered_recipients",
    "sa2_dss_carer_payment_recipients",
    "sa2_dss_youth_allowance_other_recipients",
    "sa2_dss_youth_allowance_student_recipients",
    "sa2_dss_commonwealth_rent_assistance_recipients",
)

# Convenience: block-name → column tuple.
BLOCK_COLUMNS: dict[str, tuple[str, ...]] = {
    "lag": LAG_COLUMNS,
    "upstream": UPSTREAM_COLUMNS,
    "cal": CALENDAR_COLUMNS,
    "ctx": CTX_COLUMNS,
    "stn": STN_COLUMNS,
    "wx": WX_COLUMNS,
    "sa2": SA2_COLUMNS,
}


# ---- Model variants --------------------------------------------------------

# Per spec §8.4. Order matters only for human-readability; the model treats
# the column set as unordered.
MODEL_A_BLOCKS: tuple[str, ...] = ("lag", "upstream", "cal", "ctx", "stn", "wx")
MODEL_B_BLOCKS: tuple[str, ...] = (*MODEL_A_BLOCKS, "sa2")


# ---- Categoricals ----------------------------------------------------------

# Columns LightGBM should treat as categorical (set-membership splits) rather
# than as numeric. Using the raw string brand is intentional — spec §7.5 says
# brand is exposed at multiple granularities so the model can learn
# franchisee-vs-corporate pricing differences.
CATEGORICAL_COLUMNS: frozenset[str] = frozenset(
    {
        "stn_brand_raw",
        "stn_brand_canonical",
        "wx_weather_code",
    }
)


# ---- Exclusion list --------------------------------------------------------

# Columns that exist in features.parquet but must never reach the model.
# Identifier / raw / today's-price columns. Anything in EXCLUDE_FROM_FEATURES
# would be either a leakage source (today's prices) or a useless
# high-cardinality identifier (station_id) or a non-feature key.
EXCLUDE_FROM_FEATURES: frozenset[str] = frozenset(
    {
        # Identifiers / keys
        "station_id",
        "fuel_code",
        "date",
        # Today's price (the target is t+1; today is leakage).
        "price_mean",
        "price_min",
        "price_max",
        "n_obs",
        # Targets
        "y_t1",
        "y_t1_t7",
        # Raw station fields — superseded by stn_* features.
        "name",
        "address",
        "suburb",
        "postcode",
        "brand_raw",
        "brand_canonical",
        "brand_is_major",
        "first_seen",
        "last_seen",
        # Geocoding metadata
        "lat",
        "lon",
        "geocoder",
        "mb_code",
        # SA2 keys (kept for joins; not features themselves — sa2_* are).
        "sa2_code",
        "sa2_name",
        # Counter join key, kept for traceability.
        "counter_id",
    }
)


# ---- Public API ------------------------------------------------------------


def feature_columns(
    df: pd.DataFrame, blocks: Iterable[str], *, strict: bool = True
) -> list[str]:
    """Return the list of feature columns in ``df`` for the given blocks.

    Args:
        df: a features DataFrame (columns matter; rows ignored).
        blocks: block names from ``BLOCK_COLUMNS`` keys, e.g.
            ``MODEL_A_BLOCKS`` or ``MODEL_B_BLOCKS``.
        strict: if True (default), raise if any block contains a column
            that's not present in ``df`` — the protection is "did the
            spec drift past make_features without anyone noticing?".
            If False, silently drop missing columns (useful for tests
            against trimmed synthetic frames).

    Returns:
        Ordered list of column names, suitable for ``X = df[cols]``.
        Excludes anything in ``EXCLUDE_FROM_FEATURES``.
    """
    requested: list[str] = []
    for block in blocks:
        if block not in BLOCK_COLUMNS:
            raise KeyError(
                f"unknown feature block {block!r}; expected one of {sorted(BLOCK_COLUMNS)}"
            )
        requested.extend(BLOCK_COLUMNS[block])

    if strict:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} expected feature column(s) absent from DataFrame: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                "Either spec/build drift, or pass strict=False to ignore."
            )

    return [c for c in requested if c in df.columns and c not in EXCLUDE_FROM_FEATURES]


def categorical_columns(feature_cols: Iterable[str]) -> list[str]:
    """Pick the LightGBM-categorical subset out of an arbitrary feature list.

    LightGBM accepts ``categorical_feature=`` as either column names or
    indices; we use names for readability.
    """
    return [c for c in feature_cols if c in CATEGORICAL_COLUMNS]
