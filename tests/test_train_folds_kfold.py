"""Tests for the v3.0 k-fold CV splitter (train.folds.split_kfolds).

The v2.x single-split path (split_folds) is unchanged and tested
elsewhere (test_train_folds.py). These tests only cover the new
KFoldConfig + split_kfolds.
"""
from __future__ import annotations

import datetime as dt
from itertools import pairwise

import pandas as pd
import pytest

from fuel_pred.train.folds import (
    KFOLD_NAMES,
    KFoldConfig,
    split_kfolds,
)

# --------------------------- KFoldConfig defaults ---------------------------


def test_kfold_config_defaults_match_spec_15_2() -> None:
    """The default config matches the spec §15.2 / design-doc §4.1 geometry."""
    cfg = KFoldConfig.default()
    assert cfg.k == 6
    assert cfg.test_window_months == 12
    assert cfg.val_window_days == 365
    assert cfg.gap_days == 1
    assert cfg.horizon_days == 1
    assert cfg.warmup_end == "2016-12-31"
    assert cfg.panel_end == "2026-04-30"


def test_kfold_config_fold_bounds_anchor_last_fold_to_panel_end() -> None:
    """Fold k-1 (last) tests through panel_end exactly."""
    cfg = KFoldConfig.default()
    bounds = cfg.fold_bounds()
    assert len(bounds) == cfg.k
    last_test_end = bounds[-1][-1]
    assert last_test_end == pd.Timestamp("2026-04-30")


def test_kfold_config_fold_bounds_test_windows_are_12_months() -> None:
    """Every fold's test window is exactly 12 months wide."""
    cfg = KFoldConfig.default()
    for fold_idx, (_, _, _, test_start, test_end) in enumerate(cfg.fold_bounds(), start=1):
        # Inclusive on both ends. 12 months means start+12mo-1day == end.
        expected_end = test_start + pd.DateOffset(months=12) - pd.Timedelta(days=1)
        assert test_end == expected_end, (
            f"fold {fold_idx}: test window {test_start.date()} → {test_end.date()} "
            f"is not 12 months (expected end {expected_end.date()})"
        )


def test_kfold_config_fold_bounds_test_windows_dont_overlap() -> None:
    """Sequential folds' test windows are adjacent (no overlap, no gap)."""
    cfg = KFoldConfig.default()
    bounds = cfg.fold_bounds()
    for prev, curr in pairwise(bounds):
        prev_test_end = prev[-1]
        curr_test_start = curr[-2]
        assert curr_test_start == prev_test_end + pd.Timedelta(days=1), (
            f"test windows aren't adjacent: prev ends {prev_test_end.date()}, "
            f"curr starts {curr_test_start.date()}"
        )


def test_kfold_config_fold_bounds_gap_days_separate_train_and_test() -> None:
    """gap_days days of no-mans-land between train_val_end and test_start."""
    cfg = KFoldConfig.default()  # gap_days=1
    for fold_idx, (_, _, train_val_end, test_start, _) in enumerate(cfg.fold_bounds(), start=1):
        gap = (test_start - train_val_end).days
        # gap = 1 day of separation + 1 day for the "next day after train" = 1 + gap_days
        assert gap == 1 + cfg.gap_days, (
            f"fold {fold_idx}: train_val_end={train_val_end.date()} "
            f"test_start={test_start.date()} — expected {1 + cfg.gap_days} days "
            f"between them, got {gap}"
        )


def test_kfold_config_fold_bounds_horizon_days_widens_gap() -> None:
    """horizon_days pulls the train cutoff back by (horizon_days - 1) extra
    days, so a multi-day target can't leak into the test window.

    For y_t1_t7 (horizon_days=7) the gap between train_val_end and
    test_start must be at least 7 days, so the last train row's
    mean(price[t+1..t+7]) target lands entirely in the gap.
    """
    cfg_h1 = KFoldConfig(horizon_days=1, gap_days=1)
    cfg_h7 = KFoldConfig(horizon_days=7, gap_days=1)
    for fold_idx in range(cfg_h1.k):
        _, _, tve1, ts1, _ = cfg_h1.fold_bounds()[fold_idx]
        _, _, tve7, ts7, _ = cfg_h7.fold_bounds()[fold_idx]
        gap1 = (ts1 - tve1).days
        gap7 = (ts7 - tve7).days
        # horizon=1: gap == 1 + gap_days == 2 (backward-compatible)
        assert gap1 == 1 + cfg_h1.gap_days
        # horizon=7: gap == 1 + gap_days + (7 - 1) == 8 days
        assert gap7 == 1 + cfg_h7.gap_days + (cfg_h7.horizon_days - 1), (
            f"fold {fold_idx + 1}: horizon=7 gap should be "
            f"{1 + cfg_h7.gap_days + 6}, got {gap7}"
        )
        # The 7-day target's forward reach (train_val_end + 7) must land
        # strictly before test_start (i.e. in the gap, not in test).
        assert tve7 + pd.Timedelta(days=cfg_h7.horizon_days) < ts7


def test_kfold_config_fold_bounds_train_start_after_warmup() -> None:
    """Every fold's train starts the day after warmup_end."""
    cfg = KFoldConfig.default()
    expected_train_start = pd.Timestamp("2017-01-01")
    for fold_idx, (train_start, *_) in enumerate(cfg.fold_bounds(), start=1):
        assert train_start == expected_train_start, (
            f"fold {fold_idx} train_start should be {expected_train_start.date()}, "
            f"got {train_start.date()}"
        )


def test_kfold_config_fold_bounds_val_window_is_365_days() -> None:
    """Val window is the last val_window_days of the train-val period."""
    cfg = KFoldConfig.default()
    for fold_idx, (_, val_start, train_val_end, _, _) in enumerate(cfg.fold_bounds(), start=1):
        val_span = (train_val_end - val_start).days + 1
        assert val_span == cfg.val_window_days, (
            f"fold {fold_idx}: val window {val_start.date()} → {train_val_end.date()} "
            f"is {val_span} days, expected {cfg.val_window_days}"
        )


def test_kfold_config_train_expands_across_folds() -> None:
    """Expanding-window: each fold's train_val_end >= previous fold's."""
    cfg = KFoldConfig.default()
    bounds = cfg.fold_bounds()
    for prev, curr in pairwise(bounds):
        assert curr[2] > prev[2], (
            f"train_val_end should grow each fold: prev={prev[2].date()} "
            f"curr={curr[2].date()}"
        )


def test_kfold_config_concrete_fold_6_geometry() -> None:
    """Fold 6 (last): train ends ~2025-04, test 2025-05 → 2026-04."""
    cfg = KFoldConfig.default()
    bounds = cfg.fold_bounds()
    fold_6 = bounds[-1]
    train_start, _val_start, train_val_end, test_start, test_end = fold_6
    assert train_start == pd.Timestamp("2017-01-01")
    # gap_days=1: train_val_end = test_start - 2
    assert test_start == pd.Timestamp("2025-05-01")
    assert test_end == pd.Timestamp("2026-04-30")
    assert train_val_end == pd.Timestamp("2025-04-29")  # test_start - (1 + gap_days)


# --------------------------- split_kfolds row-level ---------------------------


def _synth_panel(start: str, end: str) -> pd.DataFrame:
    """One row per date in [start, end]. Tiny — only needs a `date` column."""
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": [d.date() for d in dates]})


def test_split_kfolds_returns_k_folds() -> None:
    df = _synth_panel("2016-09-01", "2026-04-30")
    folds = split_kfolds(df)
    assert len(folds) == 6


def test_split_kfolds_each_fold_has_train_val_test() -> None:
    df = _synth_panel("2016-09-01", "2026-04-30")
    folds = split_kfolds(df)
    for i, fold in enumerate(folds, start=1):
        assert set(fold.keys()) == set(KFOLD_NAMES), f"fold {i} keys: {fold.keys()}"
        for name in KFOLD_NAMES:
            assert len(fold[name]) > 0, f"fold {i} {name} is empty"


def test_split_kfolds_no_crisis_key() -> None:
    """v3.0 drops crisis-as-separate; FoldMap entries don't contain 'crisis'."""
    df = _synth_panel("2016-09-01", "2026-04-30")
    folds = split_kfolds(df)
    for fold in folds:
        assert "crisis" not in fold


def test_split_kfolds_train_excludes_warmup_period() -> None:
    """Rows on or before warmup_end never appear in any fold's train."""
    df = _synth_panel("2016-09-01", "2026-04-30")
    folds = split_kfolds(df)
    cutoff = pd.Timestamp("2016-12-31")
    for i, fold in enumerate(folds, start=1):
        train_dates = pd.to_datetime(fold["train"]["date"])
        assert (train_dates > cutoff).all(), (
            f"fold {i} train contains warmup rows on or before {cutoff.date()}"
        )


def test_split_kfolds_train_val_disjoint() -> None:
    """Each fold's train and val have no shared (date) rows."""
    df = _synth_panel("2016-09-01", "2026-04-30")
    folds = split_kfolds(df)
    for i, fold in enumerate(folds, start=1):
        train_dates = set(pd.to_datetime(fold["train"]["date"]).dt.date)
        val_dates = set(pd.to_datetime(fold["val"]["date"]).dt.date)
        overlap = train_dates & val_dates
        assert not overlap, f"fold {i}: train ∩ val = {sorted(overlap)[:5]}..."


def test_split_kfolds_gap_day_in_no_fold() -> None:
    """The gap day(s) between train_val_end and test_start belong to neither."""
    df = _synth_panel("2016-09-01", "2026-04-30")
    cfg = KFoldConfig.default()  # gap_days=1
    folds = split_kfolds(df, kfold_config=cfg)
    bounds = cfg.fold_bounds()
    for i, (fold, b) in enumerate(zip(folds, bounds, strict=True), start=1):
        train_val_end = b[2]
        test_start = b[3]
        # Gap day(s): train_val_end+1 ... test_start-1
        for offset in range(1, cfg.gap_days + 1):
            gap_day = train_val_end + pd.Timedelta(days=offset)
            for name in KFOLD_NAMES:
                fold_dates = set(pd.to_datetime(fold[name]["date"]).dt.date)
                assert gap_day.date() not in fold_dates, (
                    f"fold {i} {name}: gap day {gap_day.date()} appears "
                    f"(test_start={test_start.date()})"
                )


def test_split_kfolds_test_windows_cover_every_date_from_2020_05() -> None:
    """Every date in [first_fold_test_start, panel_end] appears in exactly one test fold."""
    df = _synth_panel("2016-09-01", "2026-04-30")
    folds = split_kfolds(df)
    test_date_unions: dict = {}
    for i, fold in enumerate(folds, start=1):
        for d in pd.to_datetime(fold["test"]["date"]).dt.date:
            if d in test_date_unions:
                pytest.fail(f"date {d} appears in test folds {test_date_unions[d]} and {i}")
            test_date_unions[d] = i
    # Spec §15.2: fold 1 test starts 2020-05-01; fold 6 test ends 2026-04-30
    assert pd.Timestamp("2020-05-01").date() in test_date_unions
    assert pd.Timestamp("2026-04-30").date() in test_date_unions
    # 2025 + 2026 coverage required by user decision
    for d in pd.date_range("2025-01-01", "2026-04-30", freq="D"):
        assert d.date() in test_date_unions, f"date {d.date()} not covered by any test fold"


def test_split_kfolds_expanding_train_grows_each_fold() -> None:
    """Later folds have at least as many train rows as earlier folds."""
    df = _synth_panel("2016-09-01", "2026-04-30")
    folds = split_kfolds(df)
    sizes = [len(f["train"]) for f in folds]
    for prev, curr in pairwise(sizes):
        assert curr >= prev, f"train shrunk between folds: {sizes}"
    # Fold 1's train ends around 2019-04 (val carved out), fold 6's around 2024-04.
    assert sizes[-1] > sizes[0]


# --------------------------- validation + error paths ---------------------------


def test_split_kfolds_rejects_missing_date_column() -> None:
    df = pd.DataFrame({"not_date": [1, 2, 3]})
    with pytest.raises(ValueError, match="requires a 'date' column"):
        split_kfolds(df)


def test_split_kfolds_rejects_invalid_k() -> None:
    df = _synth_panel("2017-01-01", "2026-04-30")
    with pytest.raises(ValueError, match="k must be"):
        split_kfolds(df, kfold_config=KFoldConfig(k=0))


def test_split_kfolds_rejects_negative_gap_days() -> None:
    df = _synth_panel("2017-01-01", "2026-04-30")
    with pytest.raises(ValueError, match="gap_days must be"):
        split_kfolds(df, kfold_config=KFoldConfig(gap_days=-1))


def test_split_kfolds_warns_when_panel_doesnt_cover_a_fold(caplog) -> None:
    """If input data doesn't span a fold's geometry, empty slices come back with WARNING."""
    df = _synth_panel("2025-01-01", "2026-04-30")  # ~16 months total
    # k=6 with the default panel_end=2026-04-30: folds 1-4 are outside the data
    with caplog.at_level("WARNING", logger="fuel_pred.train.folds"):
        folds = split_kfolds(df, kfold_config=KFoldConfig(k=6))
    assert len(folds) == 6
    # Some folds will be empty across the board
    empty_count = sum(
        1 for f in folds if all(len(f[k]) == 0 for k in ("train", "val", "test"))
    )
    assert empty_count >= 3, f"expected >=3 fully-empty folds, got {empty_count}"
    assert any("0 rows" in rec.message for rec in caplog.records)


def test_split_kfolds_raises_on_inverted_geometry() -> None:
    """KFoldConfig with internally-inconsistent settings raises a clear error.

    Tightly impossible: val_window_days way larger than the gap between
    warmup_end and the first fold's test_start. Then val would have to
    START before warmup_end + 1, which the geometry check rejects.
    """
    df = _synth_panel("2016-09-01", "2026-04-30")
    with pytest.raises(ValueError, match="invalid geometry"):
        split_kfolds(
            df,
            kfold_config=KFoldConfig(
                # 12000 days of val + only ~9 years of data before fold 1 test:
                # val_start ends up before train_start
                k=6,
                val_window_days=10_000,
            ),
        )


# --------------------------- custom configs ---------------------------


def test_split_kfolds_honours_custom_k_and_test_window() -> None:
    """Smaller k + shorter test window for a hermetic test geometry."""
    df = _synth_panel("2016-09-01", "2026-04-30")
    cfg = KFoldConfig(
        k=3,
        test_window_months=6,
        val_window_days=180,
        gap_days=1,
        horizon_days=1,
    )
    folds = split_kfolds(df, kfold_config=cfg)
    assert len(folds) == 3
    # Each fold's test span ~6 months
    for fold in folds:
        test_dates = pd.to_datetime(fold["test"]["date"])
        span_days = (test_dates.max() - test_dates.min()).days + 1
        assert 175 <= span_days <= 185, f"test span {span_days} not ~6 months"


def test_split_kfolds_gap_days_zero_makes_train_and_test_adjacent() -> None:
    """gap_days=0 means train_val_end + 1 == test_start (re-introduces the v2.x leak)."""
    cfg = KFoldConfig(gap_days=0)
    bounds = cfg.fold_bounds()
    for _, _, train_val_end, test_start, _ in bounds:
        assert test_start == train_val_end + pd.Timedelta(days=1)


def test_split_kfolds_returns_copies() -> None:
    """Mutating a fold's slice doesn't affect the input df."""
    df = _synth_panel("2016-09-01", "2026-04-30")
    folds = split_kfolds(df)
    # Mutate fold 1 train
    if not folds[0]["train"].empty:
        folds[0]["train"].iloc[0, 0] = dt.date(1900, 1, 1)
    # Input untouched
    assert pd.to_datetime(df["date"]).min() == pd.Timestamp("2016-09-01")
