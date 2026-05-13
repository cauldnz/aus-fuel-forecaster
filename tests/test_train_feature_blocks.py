"""Tests for train.feature_blocks."""
from __future__ import annotations

import pandas as pd
import pytest

from fuel_pred.train import feature_blocks as fb

# ----------------------------- block definitions -----------------------------


def test_block_columns_match_spec_set_completely() -> None:
    """All seven §7 blocks are present in BLOCK_COLUMNS."""
    expected = {"lag", "upstream", "cal", "ctx", "stn", "wx", "sa2"}
    assert set(fb.BLOCK_COLUMNS) == expected


def test_model_a_blocks_excludes_sa2() -> None:
    """Spec §8.4: Model A is everything except sa2."""
    assert "sa2" not in fb.MODEL_A_BLOCKS


def test_model_b_blocks_is_a_plus_sa2() -> None:
    """Model B = Model A + sa2; the only difference between them."""
    assert set(fb.MODEL_B_BLOCKS) == set(fb.MODEL_A_BLOCKS) | {"sa2"}


def test_no_column_appears_in_two_blocks() -> None:
    """Each column belongs to exactly one block (no double-counting in
    feature_columns())."""
    seen: dict[str, str] = {}
    for block, cols in fb.BLOCK_COLUMNS.items():
        for col in cols:
            if col in seen:
                pytest.fail(
                    f"column {col!r} appears in both {seen[col]!r} and {block!r}"
                )
            seen[col] = block


def test_categoricals_are_a_subset_of_block_columns() -> None:
    """No categorical can refer to a column we don't actually emit."""
    all_block_cols = {c for cols in fb.BLOCK_COLUMNS.values() for c in cols}
    assert all_block_cols >= fb.CATEGORICAL_COLUMNS


def test_curated_sa2_block_excludes_high_correlation_features() -> None:
    """v2 SA2 set: the four features with |r| >= 0.5 against existing
    Model A features (per the first comparison.md) must NOT be in
    SA2_COLUMNS, but should be tracked in _DROPPED_SA2_COLUMNS so the
    audit trail survives.

    Locks in the curation decision; if someone adds them back without
    consciously updating the rationale, this test trips.
    """
    high_correlation_drops = {
        "sa2_pct_drive_to_work",
        "sa2_pct_renters",
        "sa2_motor_vehicles_per_dwelling",
        "sa2_median_age",
    }
    assert set(fb.SA2_COLUMNS).isdisjoint(high_correlation_drops), (
        "high-correlation SA2 features must not be in the curated SA2_COLUMNS"
    )
    assert set(fb._DROPPED_SA2_COLUMNS) == high_correlation_drops, (
        "_DROPPED_SA2_COLUMNS must track exactly the curation-dropped set"
    )


def test_curated_sa2_block_keeps_low_correlation_signals() -> None:
    """The two genuinely orthogonal features (|r| < 0.2 against existing
    Model A features) must remain in the curated set."""
    must_keep = {"sa2_seifa_irsd_score", "sa2_pct_employed_full_time"}
    assert must_keep <= set(fb.SA2_COLUMNS)


def test_target_columns_in_exclude_list() -> None:
    """Targets must never reach the model — guard against accidental leakage."""
    assert "y_t1" in fb.EXCLUDE_FROM_FEATURES
    assert "y_t1_t7" in fb.EXCLUDE_FROM_FEATURES


def test_todays_price_columns_in_exclude_list() -> None:
    """Today's price would leak the target — must be excluded."""
    for col in ("price_mean", "price_min", "price_max", "n_obs"):
        assert col in fb.EXCLUDE_FROM_FEATURES, f"{col!r} is leakage; must exclude"


def test_identifier_columns_in_exclude_list() -> None:
    """station_id / date / fuel_code shouldn't reach the model as features."""
    for col in ("station_id", "fuel_code", "date"):
        assert col in fb.EXCLUDE_FROM_FEATURES


# ----------------------------- feature_columns() -----------------------------


def _synthetic_df() -> pd.DataFrame:
    """Tiny DataFrame with one column from every block + targets/excludes."""
    cols: dict[str, list[object]] = {}
    for block_cols in fb.BLOCK_COLUMNS.values():
        for col in block_cols:
            cols[col] = [0.0, 1.0]
    # Targets + excludes that should be filtered out
    cols["y_t1"] = [10.0, 11.0]
    cols["y_t1_t7"] = [10.5, 11.5]
    cols["price_mean"] = [180.0, 181.0]
    cols["station_id"] = ["abc", "def"]
    cols["date"] = pd.to_datetime(["2024-01-01", "2024-01-02"])
    cols["fuel_code"] = ["U91", "U91"]
    return pd.DataFrame(cols)


def test_feature_columns_picks_model_a_set() -> None:
    """Model A picks all blocks except sa2; in our synthetic frame that's
    every block column except SA2_COLUMNS."""
    df = _synthetic_df()
    cols = fb.feature_columns(df, fb.MODEL_A_BLOCKS)
    expected_size = sum(
        len(fb.BLOCK_COLUMNS[b]) for b in fb.MODEL_A_BLOCKS
    )
    assert len(cols) == expected_size
    # No SA2 columns leaked into Model A.
    assert not any(c.startswith("sa2_") for c in cols)


def test_feature_columns_picks_model_b_set_with_sa2() -> None:
    df = _synthetic_df()
    cols = fb.feature_columns(df, fb.MODEL_B_BLOCKS)
    # B has every A column plus the sa2 set.
    cols_a = fb.feature_columns(df, fb.MODEL_A_BLOCKS)
    assert set(cols) == set(cols_a) | set(fb.SA2_COLUMNS)


def test_feature_columns_excludes_targets_even_if_user_added_target_block() -> None:
    """Defensive: even if some future block contained ``y_t1``, the
    exclude list still wins."""
    df = _synthetic_df()
    cols = fb.feature_columns(df, fb.MODEL_B_BLOCKS)
    assert "y_t1" not in cols
    assert "y_t1_t7" not in cols
    assert "price_mean" not in cols


def test_feature_columns_strict_mode_raises_on_missing() -> None:
    """If the input DataFrame is missing columns the spec promises, raise."""
    df = _synthetic_df().drop(columns=["lag_price_1"])
    with pytest.raises(ValueError, match="absent from DataFrame"):
        fb.feature_columns(df, fb.MODEL_A_BLOCKS)


def test_feature_columns_lax_mode_silently_drops_missing() -> None:
    """Tests / synthetic frames may not carry every column; lax mode
    just returns what's there."""
    df = _synthetic_df().drop(columns=["lag_price_1", "wx_temp_max_c"])
    cols = fb.feature_columns(df, fb.MODEL_A_BLOCKS, strict=False)
    assert "lag_price_1" not in cols
    assert "wx_temp_max_c" not in cols
    assert "lag_price_2" in cols  # the rest still come through


def test_feature_columns_unknown_block_raises() -> None:
    df = _synthetic_df()
    with pytest.raises(KeyError, match="unknown feature block"):
        fb.feature_columns(df, ("not_a_real_block",))


# ----------------------------- categorical_columns() -----------------------------


def test_categorical_columns_picks_subset() -> None:
    df = _synthetic_df()
    cols_b = fb.feature_columns(df, fb.MODEL_B_BLOCKS)
    cats = fb.categorical_columns(cols_b)
    # Defined categoricals: stn_brand_raw, stn_brand_canonical, wx_weather_code.
    assert set(cats) == {"stn_brand_raw", "stn_brand_canonical", "wx_weather_code"}
    # Every categorical is in the input list.
    for c in cats:
        assert c in cols_b


def test_categorical_columns_returns_empty_when_none_present() -> None:
    """Lax-mode pick with the cat columns dropped should yield no cats."""
    df = _synthetic_df().drop(
        columns=["stn_brand_raw", "stn_brand_canonical", "wx_weather_code"]
    )
    cols = fb.feature_columns(df, fb.MODEL_B_BLOCKS, strict=False)
    assert fb.categorical_columns(cols) == []
