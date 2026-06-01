"""Time-based fold splitter — two schemes.

**v2.x single-split (legacy, spec §8.3 historical):** four named folds
(``train`` / ``val`` / ``test_normal`` / ``test_crisis``). Implemented
by ``FoldConfig`` + ``split_folds``. Crisis-as-separate is deprecated
in v3.0 but the code path stays for backwards compatibility with old
training runs + tests.

**v3.0 k-fold CV (spec §15.2 default):** k expanding-window folds with
12-month test windows ending at the panel's last date. ``gap_days``
between train and test prevents the ``y_t1`` target-shift leak.
Crisis-as-separate is GONE — 2026 data rotates into the test windows
like every other year. Implemented by ``KFoldConfig`` + ``split_kfolds``.

Fold boundaries default to ``fuel_pred.config`` (which mirrors spec §8.3
for the legacy scheme and §15.2 for the k-fold scheme), overridable per
call so tests can synthesise short timelines.
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


# ============================================================
# v3.0 k-fold CV (spec §15.2) — primary scheme going forward
# ============================================================

# Key for the per-fold FoldMap entries returned by split_kfolds.
KFOLD_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class KFoldConfig:
    """Geometry of the v3.0 time-series k-fold CV (spec §15.2).

    Expanding-window chronological. Each fold's test is a fixed-width
    window of months; the windows are rolled back from ``panel_end`` so
    fold ``k`` (the last) ends at the panel boundary and every test
    date covered by the panel appears in exactly one test fold.

    Fields:

    - ``k``: number of folds (default 6 per spec §15.2)
    - ``test_window_months``: width of each test window (12 = one year)
    - ``val_window_days``: width of the val slice carved from the end
      of each fold's train (365 = last year of train used for early
      stopping; ``train`` ∩ ``val`` = ∅, ``val`` is removed from
      ``train`` before fitting)
    - ``gap_days``: days between train_end and test_start (exclusive).
      With ``horizon_days=1`` and ``gap_days=1``, last train row's
      ``y_t1`` lands on the gap day — neither train nor test. Prevents
      target leakage that the v2.x scheme has today (see spec §8.3 and
      the v3.0 Phase 1 design doc §3.1).
    - ``horizon_days``: target horizon (1 for ``y_t1``, 7 for
      ``y_t1_t7``). Used together with ``gap_days`` to compute the
      effective train cutoff.
    - ``warmup_end``: date string; rows on or before this stay in
      ``features.parquet`` for lag computation but never appear in
      any train fold. Default ``2016-12-31`` (4 months covers
      ``lag_price_28`` + ``roll_price_mean_28`` minperiods).
    - ``panel_end``: date string; last fold's test window ends here.
      Default ``2026-04-30`` (the panel's last in-range date).
    """

    k: int = 6
    test_window_months: int = 12
    val_window_days: int = 365
    gap_days: int = 1
    horizon_days: int = 1
    warmup_end: str = "2016-12-31"
    panel_end: str = "2026-04-30"

    @classmethod
    def default(cls) -> KFoldConfig:
        """Spec §15.2 6-fold geometry — the v3.0 default."""
        return cls()

    def fold_bounds(self) -> list[tuple[pd.Timestamp, ...]]:
        """Return ``[(train_start, val_start, train_val_end, test_start, test_end)]``
        for each fold, in fold order (1..k).

        - ``train_start`` = first day after ``warmup_end``
        - ``val_start`` = start of the val window inside the train period
        - ``train_val_end`` = last day of both train and val (they share
          this upper bound; ``train`` is the prefix, ``val`` is the
          ``val_window_days`` suffix)
        - ``test_start`` = ``train_val_end + 1 + gap_days``
        - ``test_end`` = inclusive end of the test window
        """
        panel_end = pd.Timestamp(self.panel_end)
        warmup_end = pd.Timestamp(self.warmup_end)
        train_start = warmup_end + pd.Timedelta(days=1)

        bounds = []
        for fold_idx in range(self.k):
            # Fold k-1 (last) ends exactly at panel_end.
            # Earlier folds back off by test_window_months each.
            months_back = (self.k - 1 - fold_idx) * self.test_window_months
            test_end = panel_end - pd.DateOffset(months=months_back)
            # +1 day at the start so the previous fold's test_end and
            # this fold's test_start don't overlap.
            test_start = (
                test_end
                - pd.DateOffset(months=self.test_window_months)
                + pd.Timedelta(days=1)
            )
            # gap_days between last train day and first test day.
            train_val_end = test_start - pd.Timedelta(days=1 + self.gap_days)
            val_start = train_val_end - pd.Timedelta(days=self.val_window_days - 1)
            bounds.append(
                (train_start, val_start, train_val_end, test_start, test_end)
            )
        return bounds


def split_kfolds(
    df: pd.DataFrame, *, kfold_config: KFoldConfig | None = None
) -> list[FoldMap]:
    """Partition ``df`` into k folds per ``kfold_config``.

    Returns a list of length ``k``; each entry is a ``FoldMap`` keyed
    by ``KFOLD_NAMES`` (``"train"``, ``"val"``, ``"test"``).

    Within each fold:
    - ``train`` and ``val`` are disjoint slices of the pre-test period:
      ``train`` is ``[warmup_end+1, val_start-1]``,
      ``val`` is ``[val_start, train_val_end]``
    - ``test`` is ``[test_start, test_end]``
    - The gap-day window ``[train_val_end+1, test_start-1]`` belongs
      to no fold (covers ``gap_days`` days; with the default
      ``gap_days=1`` that's a single date per fold)

    Args:
        df: features DataFrame; must contain a ``date`` column. Rows
            should already be filtered to U91 + non-null target before
            calling — same expectation as ``split_folds``.
        kfold_config: geometry config; defaults to ``KFoldConfig.default()``.

    Returns:
        ``[FoldMap_1, ..., FoldMap_k]``. Each FoldMap is a fresh copy.

    Logs per-fold row counts + date ranges at INFO. Catches geometry
    mistakes (empty test fold, train shrunk to 0 rows by gap, etc.).
    """
    if "date" not in df.columns:
        raise ValueError("split_kfolds requires a 'date' column in df")

    cfg = kfold_config or KFoldConfig.default()
    if cfg.k < 1:
        raise ValueError(f"k must be >= 1, got {cfg.k}")
    if cfg.gap_days < 0:
        raise ValueError(f"gap_days must be >= 0, got {cfg.gap_days}")

    dates = pd.to_datetime(df["date"])
    bounds = cfg.fold_bounds()

    folds_out: list[FoldMap] = []
    for fold_idx, bound in enumerate(bounds, start=1):
        train_start, val_start, train_val_end, test_start, test_end = bound
        # Sanity check the geometry —
        # train_start < val_start <= train_val_end < test_start <= test_end
        if not (train_start < val_start <= train_val_end < test_start <= test_end):
            raise ValueError(
                f"fold {fold_idx}: invalid geometry — "
                f"train_start={train_start.date()} val_start={val_start.date()} "
                f"train_val_end={train_val_end.date()} test_start={test_start.date()} "
                f"test_end={test_end.date()}. Check warmup_end / panel_end / k / "
                f"test_window_months / val_window_days / gap_days for consistency."
            )
        train_mask = (dates >= train_start) & (dates < val_start)
        val_mask = (dates >= val_start) & (dates <= train_val_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        fold: FoldMap = {
            "train": df.loc[train_mask].copy(),
            "val": df.loc[val_mask].copy(),
            "test": df.loc[test_mask].copy(),
        }
        for name in KFOLD_NAMES:
            slice_df = fold[name]
            if slice_df.empty:
                logger.warning(
                    "kfold fold %d/%d %s: 0 rows — check geometry against input span",
                    fold_idx, cfg.k, name,
                )
            else:
                logger.info(
                    "kfold fold %d/%d %s: %d rows (%s -> %s)",
                    fold_idx, cfg.k, name, len(slice_df),
                    pd.to_datetime(slice_df["date"]).min().date(),
                    pd.to_datetime(slice_df["date"]).max().date(),
                )
        folds_out.append(fold)

    return folds_out
