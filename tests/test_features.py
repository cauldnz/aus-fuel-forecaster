"""Tests for build.make_features.

Per CLAUDE.md, feature engineering is test-FIRST: each block has a unit test
that pins down its lag / window / null-handling behaviour. Bugs here are silent
and devastating.
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: implement once add_lag_features is filled in (Phase 4)")
def test_lag_1_is_yesterdays_price(synthetic_panel) -> None:
    pass


@pytest.mark.skip(reason="TODO")
def test_rolling_mean_uses_min_periods_window(synthetic_panel) -> None:
    """Verify no early-life leakage: rolling_mean_7 is null for first 6 days per station."""
    pass


@pytest.mark.skip(reason="TODO")
def test_target_does_not_leak(synthetic_panel) -> None:
    """y_t1 at date d must equal price_mean at date d+1, never d-1."""
    pass


@pytest.mark.skip(reason="TODO")
def test_day_of_fortnight_anchors_correctly() -> None:
    """2016-07-04 (the anchor) is day_of_fortnight = 0; 2016-07-05 is 1; 2016-07-18 is 0 again."""
    pass


@pytest.mark.skip(reason="TODO")
def test_models_a_and_b_train_on_identical_rows() -> None:
    """Per spec.md §8.4, both models must use the same training rows."""
    pass
