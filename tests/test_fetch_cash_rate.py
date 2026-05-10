"""Hermetic tests for fetch.cash_rate. RBA F1.1 endpoint mocked with `responses`."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
import responses

from fuel_pred.fetch import cash_rate


def _fake_rba_csv(rows: list[tuple[str, str]]) -> bytes:
    """Build a synthetic RBA F1.1 CSV with FIRMMCRT in column 1."""
    header_lines = [
        "F1.1 INTEREST RATES AND YIELDS - MONEY MARKET",
        "Title,Cash Rate Target,",
        "Description,RBA cash rate target; monthly average,",
        "Frequency,Monthly,",
        "Type,Original,",
        "Units,Per cent,",
        "Source,RBA,",
        "Publication date,01-May-2026,",
        "Series ID,FIRMMCRT,",
    ]
    body_lines = [f"{date},{value}," for date, value in rows]
    return ("\n".join(header_lines + body_lines) + "\n").encode("utf-8")


@pytest.fixture
def csv_payload() -> bytes:
    return _fake_rba_csv(
        [
            ("31/01/2024", "4.35"),
            ("29/02/2024", "4.35"),
            ("31/03/2024", "4.35"),
            ("30/04/2024", "4.35"),
        ]
    )


@responses.activate
def test_writes_parquet_with_expected_schema(tmp_path: Path, csv_payload: bytes) -> None:
    out = tmp_path / "cash_rate.parquet"
    responses.add(responses.GET, cash_rate.URL, body=csv_payload, status=200)

    cash_rate.fetch("2024-01-01", "2024-12-31", out)

    df = pd.read_parquet(out)
    assert list(df.columns) == ["date", "cash_rate"]
    assert len(df) == 4
    assert (df["cash_rate"] == 4.35).all()


@responses.activate
def test_filters_to_requested_range(tmp_path: Path) -> None:
    out = tmp_path / "cash_rate.parquet"
    responses.add(
        responses.GET,
        cash_rate.URL,
        body=_fake_rba_csv(
            [
                ("31/12/2023", "4.10"),
                ("31/01/2024", "4.35"),
                ("29/02/2024", "4.35"),
                ("31/12/2024", "4.10"),
            ]
        ),
        status=200,
    )
    cash_rate.fetch("2024-01-01", "2024-06-30", out)
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert df["date"].min().isoformat() == "2024-01-31"
    assert df["date"].max().isoformat() == "2024-02-29"


@responses.activate
def test_skips_when_cache_fresh(tmp_path: Path) -> None:
    out = tmp_path / "cash_rate.parquet"
    out.write_bytes(b"placeholder")
    cash_rate.fetch("2024-01-01", "2024-12-31", out, max_age_days=7.0)
    assert out.read_bytes() == b"placeholder"


@responses.activate
def test_force_bypasses_cache(tmp_path: Path, csv_payload: bytes) -> None:
    out = tmp_path / "cash_rate.parquet"
    out.write_bytes(b"placeholder")
    responses.add(responses.GET, cash_rate.URL, body=csv_payload, status=200)
    cash_rate.fetch("2024-01-01", "2024-12-31", out, force=True)
    df = pd.read_parquet(out)
    assert len(df) == 4


@responses.activate
def test_stale_cache_triggers_refetch(tmp_path: Path, csv_payload: bytes) -> None:
    import os

    out = tmp_path / "cash_rate.parquet"
    out.write_bytes(b"placeholder")
    old = time.time() - 14 * 86400
    os.utime(out, (old, old))
    responses.add(responses.GET, cash_rate.URL, body=csv_payload, status=200)
    cash_rate.fetch("2024-01-01", "2024-12-31", out, max_age_days=7.0)
    df = pd.read_parquet(out)
    assert len(df) == 4


@responses.activate
def test_empty_range_raises(tmp_path: Path, csv_payload: bytes) -> None:
    out = tmp_path / "cash_rate.parquet"
    responses.add(responses.GET, cash_rate.URL, body=csv_payload, status=200)
    with pytest.raises(RuntimeError, match="no cash rate"):
        cash_rate.fetch("2030-01-01", "2030-12-31", out)
