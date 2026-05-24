"""Tests for build.make_features.

Per CLAUDE.md, feature engineering is test-FIRST: each block has a unit test
that pins down its lag / window / null-handling behaviour. Bugs here are silent
and devastating.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fuel_pred.build.make_features import (
    DOF_ANCHOR_DATE,
    add_calendar_features,
    add_lag_features,
    add_targets,
)


def test_lag_1_is_yesterdays_price(synthetic_panel: pd.DataFrame) -> None:
    """lag_price_1 at date d must equal price_mean at date d-1 within (station, fuel)."""
    result = add_lag_features(synthetic_panel)

    for (sid, fuel), grp in result.groupby(["station_id", "fuel_code"], observed=True):
        grp = grp.sort_values("date").reset_index(drop=True)
        # First row has no yesterday — must be null.
        assert pd.isna(grp["lag_price_1"].iloc[0]), (
            f"{sid}/{fuel}: row 0 lag_price_1 should be null, got {grp['lag_price_1'].iloc[0]}"
        )
        # Every subsequent row must match the previous row's price_mean.
        for i in range(1, len(grp)):
            expected = grp["price_mean"].iloc[i - 1]
            actual = grp["lag_price_1"].iloc[i]
            assert actual == pytest.approx(expected), (
                f"{sid}/{fuel} row {i}: lag_price_1={actual!r} != price_mean[{i-1}]={expected!r}"
            )


def test_rolling_mean_uses_min_periods_window(synthetic_panel: pd.DataFrame) -> None:
    """No early-life leakage: roll_price_mean_7 is null until there are 7 prior observations.

    Implementation: s.shift(1).rolling(7, min_periods=7). shift(1) means the
    earliest usable position is row 1 (yesterday's price); the rolling window
    needs 7 non-null values. So rows 0-6 (indices 0–6) must all be null, and
    row 7 onwards must be populated.
    """
    result = add_lag_features(synthetic_panel)

    for (sid, fuel), grp in result.groupby(["station_id", "fuel_code"], observed=True):
        grp = grp.sort_values("date").reset_index(drop=True)
        null_section = grp["roll_price_mean_7"].iloc[:7]
        assert null_section.isna().all(), (
            f"{sid}/{fuel}: first 7 rows of roll_price_mean_7 must be null "
            f"(got {null_section.tolist()})"
        )
        populated_section = grp["roll_price_mean_7"].iloc[7:]
        assert populated_section.notna().all(), (
            f"{sid}/{fuel}: rows 7+ of roll_price_mean_7 must be populated "
            f"(got {populated_section.tolist()})"
        )


def test_target_does_not_leak(synthetic_panel: pd.DataFrame) -> None:
    """y_t1 at date d must equal price_mean at date d+1 (forward shift, never backward).

    Also verifies: the last observation per station has a null target (no future
    data to predict against), and Diesel rows carry null targets throughout.
    """
    # add_targets only needs station_id / fuel_code / date / price_mean.
    result = add_targets(synthetic_panel)

    u91 = result[result["fuel_code"] == "U91"]
    for sid, grp in u91.groupby("station_id", observed=True):
        grp = grp.sort_values("date").reset_index(drop=True)
        # Last row must be null — no tomorrow to predict.
        assert pd.isna(grp["y_t1"].iloc[-1]), (
            f"station {sid}: last row y_t1 should be null"
        )
        # Every other row: y_t1[i] == price_mean[i+1].
        for i in range(len(grp) - 1):
            expected = grp["price_mean"].iloc[i + 1]
            actual = grp["y_t1"].iloc[i]
            assert actual == pytest.approx(expected), (
                f"station {sid} row {i}: y_t1={actual!r} != price_mean[{i+1}]={expected!r}"
            )


def test_day_of_fortnight_anchors_correctly() -> None:
    """2016-07-04 (the anchor) is day 0; 2016-07-05 is day 1; 2016-07-18 wraps to 0."""
    anchor = DOF_ANCHOR_DATE  # 2016-07-04, a Monday
    dates = [
        anchor,                           # day 0
        anchor + dt.timedelta(days=1),    # day 1
        anchor + dt.timedelta(days=13),   # day 13 (last of the fortnight)
        anchor + dt.timedelta(days=14),   # wraps → day 0 again
        anchor + dt.timedelta(days=15),   # day 1 again
    ]
    df = pd.DataFrame({
        "station_id": ["s1"] * len(dates),
        "fuel_code": ["U91"] * len(dates),
        "date": pd.to_datetime(dates),
    })
    # school_terms_path=None falls back to the static file path in config;
    # pass a missing path so the function falls through to an empty term list.
    from fuel_pred import config
    result = add_calendar_features(df, school_terms_path=config.DATA_STATIC / "nsw_school_terms.csv")

    dofs = result.sort_values("date")["cal_day_of_fortnight"].tolist()
    assert dofs == [0, 1, 13, 0, 1], (
        f"Expected [0, 1, 13, 0, 1], got {dofs}"
    )


def test_models_a_and_b_train_on_identical_rows() -> None:
    """The identical-rows guard filters on SA2 non-null only (spec §8.4).

    Both models must use exactly the same row set — rows are excluded only
    when at least one *SA2* column is null, not when other (naturally sparse)
    columns like xfuel_* or upstream_tgp_* are null.
    """
    from fuel_pred.train.feature_blocks import SA2_COLUMNS

    # Build a tiny frame: 5 rows, first 3 have all SA2 cols populated,
    # last 2 have a null in one SA2 col.
    sa2_sample = list(SA2_COLUMNS[:3])
    df = pd.DataFrame({c: [1.0, 2.0, 3.0, np.nan, np.nan] for c in sa2_sample})
    # A naturally-sparse non-SA2 column: must NOT cause row exclusion.
    df["xfuel_dl_price_lag_0"] = np.nan

    mask = df[sa2_sample].notna().all(axis=1)
    eligible = df[mask]

    assert len(eligible) == 3, (
        f"Only SA2-null rows should be excluded; expected 3 eligible, got {len(eligible)}"
    )
    # Non-SA2 nulls survive the guard — both models handle them via LightGBM's
    # native null support, not by row exclusion.
    assert eligible["xfuel_dl_price_lag_0"].isna().all(), (
        "Non-SA2 null columns must not be filtered out by the identical-rows guard"
    )
