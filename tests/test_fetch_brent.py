"""Hermetic tests for fetch.brent.

We mock `yfinance.Ticker` rather than going over the wire — yfinance issues
HTTP requests during construction in some versions, and the public surface
we depend on (`.history(...)`) is stable enough to mock cleanly.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fuel_pred.fetch import brent


def _fake_history_frame(n_days: int = 5) -> pd.DataFrame:
    """A frame shaped like yfinance's `Ticker.history(...)` output.

    Index is a tz-aware DatetimeIndex (`Date`), columns include OHLCV
    plus a couple of extras that real yfinance returns ('Dividends',
    'Stock Splits') so we can prove the fetcher trims to the spec'd schema.
    """
    idx = pd.date_range("2024-01-02", periods=n_days, freq="D", tz="America/New_York", name="Date")
    return pd.DataFrame(
        {
            "Open": [80.0 + i for i in range(n_days)],
            "High": [81.0 + i for i in range(n_days)],
            "Low": [79.0 + i for i in range(n_days)],
            "Close": [80.5 + i for i in range(n_days)],
            "Volume": [1000 + i for i in range(n_days)],
            "Dividends": [0.0] * n_days,
            "Stock Splits": [0.0] * n_days,
        },
        index=idx,
    )


def test_writes_parquet_with_expected_schema(tmp_path: Path) -> None:
    out = tmp_path / "brent.parquet"
    fake = MagicMock()
    fake.history.return_value = _fake_history_frame()

    with patch.object(brent.yf, "Ticker", return_value=fake) as ticker_cls:
        brent.fetch("2024-01-02", "2024-01-07", out)

    ticker_cls.assert_called_once_with("BZ=F")
    assert out.exists()

    df = pd.read_parquet(out)
    assert list(df.columns) == list(brent.SCHEMA)
    assert len(df) == 5
    # Date column is plain `date`, not datetime / tz-aware.
    import datetime as dt

    assert isinstance(df["date"].iloc[0], dt.date)
    assert not isinstance(df["date"].iloc[0], dt.datetime)


def test_skips_fetch_when_cache_fresh(tmp_path: Path) -> None:
    out = tmp_path / "brent.parquet"
    out.write_bytes(b"placeholder")

    with patch.object(brent.yf, "Ticker") as ticker_cls:
        brent.fetch("2024-01-02", "2024-01-07", out, max_age_days=1.0)

    ticker_cls.assert_not_called()
    # File was not overwritten.
    assert out.read_bytes() == b"placeholder"


def test_force_bypasses_cache(tmp_path: Path) -> None:
    out = tmp_path / "brent.parquet"
    out.write_bytes(b"placeholder")

    fake = MagicMock()
    fake.history.return_value = _fake_history_frame(n_days=2)

    with patch.object(brent.yf, "Ticker", return_value=fake):
        brent.fetch("2024-01-02", "2024-01-04", out, force=True)

    df = pd.read_parquet(out)
    assert len(df) == 2


def test_stale_cache_triggers_refetch(tmp_path: Path) -> None:
    out = tmp_path / "brent.parquet"
    out.write_bytes(b"placeholder")
    # Backdate the file so it counts as stale.
    old = time.time() - 5 * 86400
    import os

    os.utime(out, (old, old))

    fake = MagicMock()
    fake.history.return_value = _fake_history_frame(n_days=3)

    with patch.object(brent.yf, "Ticker", return_value=fake):
        brent.fetch("2024-01-02", "2024-01-05", out, max_age_days=1.0)

    df = pd.read_parquet(out)
    assert len(df) == 3


def test_empty_response_raises(tmp_path: Path) -> None:
    out = tmp_path / "brent.parquet"
    fake = MagicMock()
    fake.history.return_value = pd.DataFrame()

    with patch.object(brent.yf, "Ticker", return_value=fake), pytest.raises(RuntimeError):
        brent.fetch("2024-01-02", "2024-01-07", out)

    assert not out.exists()
