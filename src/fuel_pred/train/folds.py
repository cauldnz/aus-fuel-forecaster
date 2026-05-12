"""Time-based train/val/test/test_crisis splitter (spec §8.3).

No k-fold CV in v1 — a single time-based holdout is the validation. The
2026 fuel crisis is held out as a separate "test_crisis" fold so the
headline metrics on the "test_normal" fold remain comparable to a
pre-crisis baseline (spec §11).

Fold boundaries default to ``fuel_pred.config`` (which mirrors spec §8.3),
overridable per call so tests can synthesise short timelines.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from fuel_pred import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldConfig:
    """Boundaries of the four time-based folds.

    All four are inclusive of their start and end. The complement —
    rows whose ``date`` falls outside every defined window — is
    silently dropped (typical case: rows before ``train_start``
    because lag warmup hasn't completed, or rows on a date precisely
    inside a stated gap).
    """

    train_start: str = config.SPAN_START
    train_end: str = config.TRAIN_END
    val_start: str = config.VAL_START
    val_end: str = config.VAL_END
    test_normal_start: str = config.TEST_START
    test_normal_end: str = config.TEST_NORMAL_END
    test_crisis_start: str = config.TEST_CRISIS_START
    test_crisis_end: str | None = None  # ``None`` = end of data

    @classmethod
    def default(cls) -> FoldConfig:
        """The spec §8.3 boundaries baked into ``fuel_pred.config``."""
        return cls()


# Type alias for the splitter's output. A dict so callers can address
# folds by name (``folds["train"]``) rather than positional indexing.
FoldMap = dict[str, pd.DataFrame]

# Fixed key order — used by callers that want to enumerate or table-format.
FOLD_NAMES: tuple[str, ...] = ("train", "val", "test_normal", "test_crisis")


def split_folds(df: pd.DataFrame, *, fold: FoldConfig | None = None) -> FoldMap:
    """Partition ``df`` into the four spec §8.3 time-based folds.

    Args:
        df: a features DataFrame; must contain a ``date`` column. Rows
            should already be filtered to U91 + non-null target before
            calling — this function does NOT do that filtering.
        fold: boundary config; defaults to ``FoldConfig.default()``.

    Returns:
        ``{"train": ..., "val": ..., "test_normal": ..., "test_crisis": ...}``.
        Each value is a copy (so downstream mutation can't leak across folds).

    Logs:
        Per-fold row count + date range at INFO. Helps catch fold-config
        mistakes (e.g. an empty val fold because dates were misformatted).
    """
    if "date" not in df.columns:
        raise ValueError("split_folds requires a 'date' column in df")

    fold = fold or FoldConfig.default()
    dates = pd.to_datetime(df["date"])

    train_start = pd.Timestamp(fold.train_start)
    train_end = pd.Timestamp(fold.train_end)
    val_start = pd.Timestamp(fold.val_start)
    val_end = pd.Timestamp(fold.val_end)
    test_normal_start = pd.Timestamp(fold.test_normal_start)
    test_normal_end = pd.Timestamp(fold.test_normal_end)
    test_crisis_start = pd.Timestamp(fold.test_crisis_start)
    test_crisis_end = (
        pd.Timestamp(fold.test_crisis_end) if fold.test_crisis_end is not None else None
    )

    if not (train_end < val_start <= val_end < test_normal_start
            <= test_normal_end < test_crisis_start):
        raise ValueError(
            "fold boundaries must be strictly chronological: "
            f"train_end={train_end.date()} < val_start={val_start.date()} ≤ "
            f"val_end={val_end.date()} < test_normal_start={test_normal_start.date()} ≤ "
            f"test_normal_end={test_normal_end.date()} < "
            f"test_crisis_start={test_crisis_start.date()}"
        )

    masks: dict[str, pd.Series] = {
        "train": (dates >= train_start) & (dates <= train_end),
        "val": (dates >= val_start) & (dates <= val_end),
        "test_normal": (dates >= test_normal_start) & (dates <= test_normal_end),
        "test_crisis": (
            (dates >= test_crisis_start)
            & (
                (dates <= test_crisis_end)
                if test_crisis_end is not None
                else pd.Series(True, index=dates.index)
            )
        ),
    }

    out: FoldMap = {}
    for name in FOLD_NAMES:
        mask = masks[name]
        slice_df = df.loc[mask].copy()
        out[name] = slice_df
        if slice_df.empty:
            logger.warning("fold %s: 0 rows — check boundary config + input data", name)
        else:
            logger.info(
                "fold %s: %d rows (%s -> %s)",
                name,
                len(slice_df),
                pd.to_datetime(slice_df["date"]).min().date(),
                pd.to_datetime(slice_df["date"]).max().date(),
            )

    return out
