"""Tests for train.feature_blocks."""
from __future__ import annotations

import pandas as pd
import pytest

from fuel_pred.train import feature_blocks as fb

# ----------------------------- block definitions -----------------------------


def test_block_columns_match_spec_set_completely() -> None:
    """All seven §7 blocks + the §13.6 Phase 1 venue + §13.7 v2.0 wx_gfs block."""
    expected = {"lag", "upstream", "cal", "ctx", "stn", "wx", "sa2", "venue", "wx_gfs"}
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
    # Model B doesn't include the venue block, so stn_nearest_venue_type
    # isn't expected here. Defined categoricals for Model B:
    # stn_brand_raw, stn_brand_canonical, wx_weather_code.
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


def test_model_b_prime_blocks_is_b_plus_venue() -> None:
    """Spec §13.6 Phase 1: Model B' = Model B + venue block."""
    assert set(fb.MODEL_B_PRIME_BLOCKS) == set(fb.MODEL_B_BLOCKS) | {"venue"}


def test_venue_block_contains_expected_columns() -> None:
    """VENUE_COLUMNS includes the 4 station-side + the long-weekend cal flag."""
    assert set(fb.VENUE_COLUMNS) == {
        "stn_nearest_venue_km",
        "stn_nearest_venue_capacity",
        "stn_nearest_venue_type",
        "stn_n_venues_within_5km",
        "cal_is_pre_long_weekend",
    }


def test_categorical_columns_picks_venue_type_in_model_b_prime() -> None:
    """When Model B' is the requested set, stn_nearest_venue_type is categorical."""
    df = _synthetic_df()
    cols_bp = fb.feature_columns(df, fb.MODEL_B_PRIME_BLOCKS)
    cats = fb.categorical_columns(cols_bp)
    assert "stn_nearest_venue_type" in cats


# ---------- GFS weather-block variants (spec §13.7 v2.0) -----------------


def test_model_a_gfs_blocks_swaps_wx_for_wx_gfs() -> None:
    """Model A GFS variant: identical to Model A but the wx block becomes wx_gfs."""
    assert set(fb.MODEL_A_GFS_BLOCKS) == set(fb.MODEL_A_BLOCKS) - {"wx"} | {"wx_gfs"}


def test_model_b_gfs_blocks_swaps_wx_for_wx_gfs() -> None:
    """Model B GFS variant: identical to Model B but the wx block becomes wx_gfs."""
    assert set(fb.MODEL_B_GFS_BLOCKS) == set(fb.MODEL_B_BLOCKS) - {"wx"} | {"wx_gfs"}


def test_wx_gfs_block_in_block_columns() -> None:
    """BLOCK_COLUMNS must map 'wx_gfs' to WX_COLUMNS_GFS_T1, the day-1 set only."""
    assert fb.BLOCK_COLUMNS["wx_gfs"] == fb.WX_COLUMNS_GFS_T1
    assert len(fb.BLOCK_COLUMNS["wx_gfs"]) == 5  # 5 wx_* vars, day-1 only


def test_wx_weather_code_t1_is_categorical() -> None:
    """The day-1 GFS weather code is categorical (spec §13.7 v2.0).

    GFS/GEFS doesn't emit WMO codes — this is null today — but listing
    it as categorical means LightGBM treats the column type correctly
    if a derivation lands later. Session 3 flagged the missing suffixed
    entry as a future-pitfall; Session 4a fixes it.
    Wider _t2..t7 horizons are added in v2.1 alongside the multi-horizon
    block; they're deliberately NOT in CATEGORICAL_COLUMNS yet because
    they're not in any BLOCK_COLUMNS entry (subset invariant).
    """
    assert "wx_weather_code_t1" in fb.CATEGORICAL_COLUMNS


def test_wx_gfs_block_excludes_multi_horizon_columns() -> None:
    """wx_gfs is t1-only; _t2..t7 cols must NOT be in MODEL_*_GFS_BLOCKS expansion."""
    # All cols implied by MODEL_B_GFS_BLOCKS
    expanded = set()
    for block in fb.MODEL_B_GFS_BLOCKS:
        expanded.update(fb.BLOCK_COLUMNS[block])
    # No _t2..t7 weather columns should be present
    for h in range(2, 8):
        for var in ("temp_max_c", "temp_min_c", "precipitation_mm",
                    "wind_speed_max_kmh", "weather_code"):
            assert f"wx_{var}_t{h}" not in expanded, (
                f"wx_{var}_t{h} should not be in v2.0 GFS model — "
                f"only t1 horizon for spec §13.7"
            )
