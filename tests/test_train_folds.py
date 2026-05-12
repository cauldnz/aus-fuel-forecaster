"""Tests for train.folds."""
from __future__ import annotations

import pandas as pd
import pytest

from fuel_pred.train import folds as f


def _date_range_df(start: str, end: str) -> pd.DataFrame:
    """One row per day between start..end inclusive."""
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": dates, "y_t1": range(len(dates))})


def test_default_fold_config_matches_config_constants() -> None:
    """The default boundaries come straight from fuel_pred.config."""
    from fuel_pred import config

    fc = f.FoldConfig.default()
    assert fc.train_start == config.SPAN_START
    assert fc.train_end == config.TRAIN_END
    assert fc.val_start == config.VAL_START
    assert fc.val_end == config.VAL_END
    assert fc.test_normal_start == config.TEST_START
    assert fc.test_normal_end == config.TEST_NORMAL_END
    assert fc.test_crisis_start == config.TEST_CRISIS_START


def test_split_folds_row_count_invariant() -> None:
    """Sum of fold sizes equals input size — no row lost or double-counted."""
    df = _date_range_df("2016-09-01", "2026-04-30")
    folds = f.split_folds(df)
    total = sum(len(v) for v in folds.values())
    assert total == len(df), (
        f"fold sum {total} != input size {len(df)} — "
        "rows dropped or duplicated across boundaries"
    )


def test_split_folds_no_overlap_between_folds() -> None:
    """The four fold date ranges must not share any single date."""
    df = _date_range_df("2016-09-01", "2026-04-30")
    folds = f.split_folds(df)
    seen_dates: set[pd.Timestamp] = set()
    for name, slice_df in folds.items():
        dates = set(pd.to_datetime(slice_df["date"]).tolist())
        overlap = seen_dates & dates
        assert not overlap, (
            f"fold {name!r} overlaps with previous folds on {len(overlap)} dates"
        )
        seen_dates |= dates


def test_split_folds_train_ends_on_train_end_inclusive() -> None:
    df = _date_range_df("2016-09-01", "2026-04-30")
    folds = f.split_folds(df)
    train_max = pd.to_datetime(folds["train"]["date"]).max()
    assert train_max == pd.Timestamp("2022-12-31")


def test_split_folds_test_crisis_starts_on_test_crisis_start_inclusive() -> None:
    df = _date_range_df("2016-09-01", "2026-04-30")
    folds = f.split_folds(df)
    crisis_min = pd.to_datetime(folds["test_crisis"]["date"]).min()
    assert crisis_min == pd.Timestamp("2026-01-01")


def test_split_folds_returns_copies_not_views() -> None:
    """Mutating one fold must not bleed into the input."""
    df = _date_range_df("2024-01-01", "2024-01-10")
    fold = f.FoldConfig(
        train_start="2024-01-01",
        train_end="2024-01-03",
        val_start="2024-01-04",
        val_end="2024-01-05",
        test_normal_start="2024-01-06",
        test_normal_end="2024-01-08",
        test_crisis_start="2024-01-09",
    )
    folds = f.split_folds(df, fold=fold)
    # Mutate a fold; original should be untouched.
    folds["train"].loc[folds["train"].index[0], "y_t1"] = 99999
    assert df["y_t1"].iloc[0] != 99999, "fold mutation leaked back to input"


def test_split_folds_raises_on_missing_date_column() -> None:
    df = pd.DataFrame({"y": [1, 2, 3]})
    with pytest.raises(ValueError, match="'date' column"):
        f.split_folds(df)


def test_split_folds_raises_on_non_chronological_boundaries() -> None:
    bad_fold = f.FoldConfig(
        train_start="2024-01-01",
        train_end="2024-06-30",
        val_start="2024-05-01",  # BEFORE train_end → invalid
        val_end="2024-06-30",
        test_normal_start="2024-07-01",
        test_normal_end="2024-12-31",
        test_crisis_start="2025-01-01",
    )
    df = _date_range_df("2024-01-01", "2024-12-31")
    with pytest.raises(ValueError, match="strictly chronological"):
        f.split_folds(df, fold=bad_fold)


def test_split_folds_test_crisis_end_clips_when_provided() -> None:
    """Optional test_crisis_end clips the open-ended crisis fold."""
    df = _date_range_df("2026-01-01", "2026-12-31")
    fold = f.FoldConfig(
        train_start="2024-01-01",
        train_end="2024-12-31",
        val_start="2025-01-01",
        val_end="2025-06-30",
        test_normal_start="2025-07-01",
        test_normal_end="2025-12-31",
        test_crisis_start="2026-01-01",
        test_crisis_end="2026-06-30",
    )
    folds = f.split_folds(df, fold=fold)
    crisis_max = pd.to_datetime(folds["test_crisis"]["date"]).max()
    assert crisis_max == pd.Timestamp("2026-06-30")
    # Rows after 2026-06-30 are dropped from every fold (no overlap).
    total = sum(len(v) for v in folds.values())
    assert total < len(df), "test_crisis_end should clip"


def test_split_folds_warns_on_empty_fold(caplog: pytest.LogCaptureFixture) -> None:
    """Empty folds are likely a config bug — surface them via WARNING."""
    df = _date_range_df("2024-01-01", "2024-12-31")
    fold = f.FoldConfig(
        train_start="2024-01-01",
        train_end="2024-06-30",
        val_start="2024-07-01",
        val_end="2024-08-31",
        test_normal_start="2024-09-01",
        test_normal_end="2024-12-31",
        test_crisis_start="2025-01-01",  # well past the input data
    )
    with caplog.at_level("WARNING", logger="fuel_pred.train.folds"):
        folds = f.split_folds(df, fold=fold)
    assert len(folds["test_crisis"]) == 0
    assert any("test_crisis" in r.message and "0 rows" in r.message for r in caplog.records)


def test_fold_names_constant_matches_dict_keys() -> None:
    """FOLD_NAMES is the documented enumeration order — must match the
    keys split_folds() returns, lest downstream callers iterate in a
    different order than the splitter populates."""
    df = _date_range_df("2016-09-01", "2026-04-30")
    folds = f.split_folds(df)
    assert set(folds.keys()) == set(f.FOLD_NAMES)
