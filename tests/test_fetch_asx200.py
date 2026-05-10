"""Hermetic tests for fetch.asx200 (yfinance ^AXJO)."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fuel_pred.fetch import asx200


def _fake_history_frame(n_days: int = 5) -> pd.DataFrame:
    """A frame shaped like yfinance's `Ticker.history(...)` output."""
    idx = pd.date_range(
        "2024-01-02", periods=n_days, freq="D", tz="Australia/Sydney", name="Date"
    )
    return pd.DataFrame(
        {
            "Open": [7400.0 + i for i in range(n_days)],
            "High": [7450.0 + i for i in range(n_days)],
            "Low": [7350.0 + i for i in range(n_days)],
            "Close": [7420.0 + i for i in range(n_days)],
            "Volume": [1_000_000 + i for i in range(n_days)],
            "Dividends": [0.0] * n_days,
            "Stock Splits": [0.0] * n_days,
        },
        index=idx,
    )


def test_writes_parquet_with_expected_schema(tmp_path: Path) -> None:
    out = tmp_path / "asx200.parquet"
    fake = MagicMock()
    fake.history.return_value = _fake_history_frame()

    with patch.object(asx200.yf, "Ticker", return_value=fake) as ticker_cls:
        asx200.fetch("2024-01-02", "2024-01-07", out)

    ticker_cls.assert_called_once_with("^AXJO")
    df = pd.read_parquet(out)
    assert list(df.columns) == list(asx200.SCHEMA)
    assert len(df) == 5
    import datetime as dt

    assert isinstance(df["date"].iloc[0], dt.date)
    assert not isinstance(df["date"].iloc[0], dt.datetime)


def test_skips_when_cache_fresh(tmp_path: Path) -> None:
    out = tmp_path / "asx200.parquet"
    out.write_bytes(b"placeholder")
    with patch.object(asx200.yf, "Ticker") as ticker_cls:
        asx200.fetch("2024-01-02", "2024-01-07", out, max_age_days=1.0)
    ticker_cls.assert_not_called()


def test_force_bypasses_cache(tmp_path: Path) -> None:
    out = tmp_path / "asx200.parquet"
    out.write_bytes(b"placeholder")
    fake = MagicMock()
    fake.history.return_value = _fake_history_frame(n_days=2)
    with patch.object(asx200.yf, "Ticker", return_value=fake):
        asx200.fetch("2024-01-02", "2024-01-04", out, force=True)
    df = pd.read_parquet(out)
    assert len(df) == 2


def test_stale_cache_triggers_refetch(tmp_path: Path) -> None:
    import os

    out = tmp_path / "asx200.parquet"
    out.write_bytes(b"placeholder")
    old = time.time() - 5 * 86400
    os.utime(out, (old, old))

    fake = MagicMock()
    fake.history.return_value = _fake_history_frame(n_days=3)

    with patch.object(asx200.yf, "Ticker", return_value=fake):
        asx200.fetch("2024-01-02", "2024-01-05", out, max_age_days=1.0)

    df = pd.read_parquet(out)
    assert len(df) == 3


def test_empty_response_raises(tmp_path: Path) -> None:
    out = tmp_path / "asx200.parquet"
    fake = MagicMock()
    fake.history.return_value = pd.DataFrame()
    with patch.object(asx200.yf, "Ticker", return_value=fake), pytest.raises(RuntimeError):
        asx200.fetch("2024-01-02", "2024-01-07", out)
    assert not out.exists()
