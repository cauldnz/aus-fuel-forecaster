"""Shared pytest fixtures.

Per CLAUDE.md, all tests in this directory must be hermetic. Fixtures here
provide synthetic data and mocked HTTP — no real network calls.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _disable_tenacity_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make tenacity retries instant.

    The ``@retry`` decorator captures ``wait_exponential`` at import time, so
    patching the symbol is too late. tenacity defaults to ``time.sleep`` for
    waiting between attempts; replacing it here keeps real-failure tests
    instant without changing behaviour under a happy-path mock.
    """
    import time as time_module

    monkeypatch.setattr(time_module, "sleep", lambda *_a, **_kw: None)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """A tmp dir with the standard data/raw, data/interim, data/processed layout."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "interim").mkdir()
    (tmp_path / "processed").mkdir()
    return tmp_path


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """A tiny synthetic (station × fuel × date) panel for feature tests.

    Two stations, one fuel, 30 days. Use this to test lag / rolling / target
    construction in isolation without the rest of the pipeline.
    """
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    stations = ["s1", "s2"]
    rows = []
    for station in stations:
        for i, d in enumerate(dates):
            base = 180 if station == "s1" else 195
            rows.append(
                {
                    "station_id": station,
                    "fuel_code": "U91",
                    "date": d,
                    "price_mean": base + (i % 7),  # weekly cycle
                    "price_min": base + (i % 7) - 1,
                    "price_max": base + (i % 7) + 1,
                    "n_obs": 3,
                }
            )
    return pd.DataFrame(rows)
