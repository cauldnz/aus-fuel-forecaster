"""Hermetic tests for fetch.inflation_expectations. RBA G3 mocked with `responses`."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
import responses

from fuel_pred.fetch import inflation_expectations as ie


def _fake_g3_csv(rows: list[tuple[str, str]]) -> bytes:
    """Build a synthetic RBA G3 CSV with GCONEXP in column 1."""
    header_lines = [
        "G3 INFLATION EXPECTATIONS",
        "Title,Consumer inflation expectations - 1-year ahead,",
        "Description,Survey measure of consumer inflation expectations,",
        "Frequency,Quarterly,",
        "Type,Original,",
        "Units,Per cent,",
        "Source,MI,",
        "Publication date,19-Mar-2026,",
        "Series ID,GCONEXP,",
    ]
    body_lines = [f"{date},{value}," for date, value in rows]
    return ("\n".join(header_lines + body_lines) + "\n").encode("utf-8")


@pytest.fixture
def csv_payload() -> bytes:
    return _fake_g3_csv(
        [
            ("31/03/2024", "4.5"),
            ("30/06/2024", "4.2"),
            ("30/09/2024", "3.9"),
            ("31/12/2024", "3.7"),
        ]
    )


@responses.activate
def test_writes_parquet_with_expected_schema(tmp_path: Path, csv_payload: bytes) -> None:
    out = tmp_path / "ie.parquet"
    responses.add(responses.GET, ie.URL, body=csv_payload, status=200)

    ie.fetch("2024-01-01", "2024-12-31", out)

    df = pd.read_parquet(out)
    assert list(df.columns) == ["date", "inflation_expectations"]
    assert len(df) == 4
    assert df["inflation_expectations"].iloc[0] == pytest.approx(4.5)
    assert df["inflation_expectations"].iloc[-1] == pytest.approx(3.7)


@responses.activate
def test_filters_to_requested_range(tmp_path: Path) -> None:
    out = tmp_path / "ie.parquet"
    responses.add(
        responses.GET,
        ie.URL,
        body=_fake_g3_csv(
            [
                ("31/12/2023", "5.0"),
                ("31/03/2024", "4.5"),
                ("30/06/2024", "4.2"),
                ("31/12/2024", "3.7"),
            ]
        ),
        status=200,
    )
    ie.fetch("2024-01-01", "2024-09-30", out)
    df = pd.read_parquet(out)
    assert len(df) == 2  # Q1 + Q2 only


@responses.activate
def test_skips_when_cache_fresh(tmp_path: Path) -> None:
    out = tmp_path / "ie.parquet"
    out.write_bytes(b"placeholder")
    ie.fetch("2024-01-01", "2024-12-31", out, max_age_days=14.0)
    assert out.read_bytes() == b"placeholder"


@responses.activate
def test_force_bypasses_cache(tmp_path: Path, csv_payload: bytes) -> None:
    out = tmp_path / "ie.parquet"
    out.write_bytes(b"placeholder")
    responses.add(responses.GET, ie.URL, body=csv_payload, status=200)
    ie.fetch("2024-01-01", "2024-12-31", out, force=True)
    df = pd.read_parquet(out)
    assert len(df) == 4


@responses.activate
def test_stale_cache_triggers_refetch(tmp_path: Path, csv_payload: bytes) -> None:
    import os

    out = tmp_path / "ie.parquet"
    out.write_bytes(b"placeholder")
    old = time.time() - 30 * 86400
    os.utime(out, (old, old))
    responses.add(responses.GET, ie.URL, body=csv_payload, status=200)
    ie.fetch("2024-01-01", "2024-12-31", out, max_age_days=14.0)
    df = pd.read_parquet(out)
    assert len(df) == 4


@responses.activate
def test_empty_range_raises(tmp_path: Path, csv_payload: bytes) -> None:
    out = tmp_path / "ie.parquet"
    responses.add(responses.GET, ie.URL, body=csv_payload, status=200)
    with pytest.raises(RuntimeError, match="no inflation-expectations"):
        ie.fetch("2030-01-01", "2030-12-31", out)
