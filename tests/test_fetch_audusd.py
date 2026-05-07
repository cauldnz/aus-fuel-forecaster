"""Hermetic tests for fetch.audusd. RBA F11.1 endpoints mocked with `responses`."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
import responses

from fuel_pred.fetch import audusd


def _fake_rba_csv(rows: list[tuple[str, str, str]]) -> bytes:
    """Build a synthetic RBA F11.1 CSV.

    Each `rows` tuple is (date, audusd, gbp) — gbp is included to prove that
    the parser locates the correct column when multiple series are present.
    """
    header_lines = [
        "Title,Daily Foreign Exchange Rates,,",
        ",,,",
        "Description,US dollar,UK pound sterling,",
        "Frequency,Daily,Daily,",
        "Type,Original,Original,",
        "Units,USD per AUD,GBP per AUD,",
        "Source,Reserve Bank of Australia,Reserve Bank of Australia,",
        "Publication date,2026-05-01,2026-05-01,",
        "Series ID,FXRUSD,FXRUKPS,",
    ]
    body_lines = [f"{d},{aud},{gbp}," for d, aud, gbp in rows]
    return ("\n".join(header_lines + body_lines) + "\n").encode("utf-8")


def _register_all_csv_sources(rows_per_url: dict[str, bytes]) -> None:
    """Register a CSV body for every URL in ``audusd.SOURCES``.

    The real URL list mixes CSV and XLS; in tests we serve CSV at every URL
    and patch the format detection to treat them all as CSV.
    """
    for url, body in rows_per_url.items():
        responses.add(responses.GET, url, body=body, status=200)


@pytest.fixture(autouse=True)
def _force_csv_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """In tests we serve all RBA payloads as CSV; bypass the .xls branch."""
    original = audusd._read_rba_table

    def _always_csv(payload: bytes, url: str) -> pd.DataFrame:
        return original(payload, url + ".csv" if not url.lower().endswith(".csv") else url)

    monkeypatch.setattr(audusd, "_read_rba_table", _always_csv)


@pytest.fixture
def hist_2014_2017_csv() -> bytes:
    return _fake_rba_csv(
        [
            ("01-Dec-2017", "0.7600", "0.5500"),
            ("31-Dec-2017", "0.7800", "0.5800"),
        ]
    )


@pytest.fixture
def hist_2018_2022_csv() -> bytes:
    return _fake_rba_csv(
        [
            ("02-Jan-2018", "0.7820", "0.5810"),
            # Overlap on 31-Dec-2017 — the later file wins on dedupe.
            ("31-Dec-2017", "0.9999", "0.5499"),
            ("01-Jan-2022", "", ""),  # blank — must drop
        ]
    )


@pytest.fixture
def current_csv() -> bytes:
    return _fake_rba_csv(
        [
            ("03-Jan-2024", "0.6750", "0.5300"),
            ("04-Jan-2024", "0.6800", "0.5310"),
        ]
    )


def _all_sources(
    h1: bytes, h2: bytes, c: bytes
) -> dict[str, bytes]:
    return {
        audusd.URL_HIST_2014_2017: h1,
        audusd.URL_HIST_2018_2022: h2,
        audusd.URL_CURRENT: c,
    }


@responses.activate
def test_fetch_writes_parquet_with_expected_schema(
    tmp_path: Path,
    hist_2014_2017_csv: bytes,
    hist_2018_2022_csv: bytes,
    current_csv: bytes,
) -> None:
    out = tmp_path / "audusd.parquet"
    _register_all_csv_sources(_all_sources(hist_2014_2017_csv, hist_2018_2022_csv, current_csv))

    audusd.fetch("2014-01-01", "2024-12-31", out)

    df = pd.read_parquet(out)
    assert list(df.columns) == ["date", "audusd"]
    # 5 distinct trading days; the blank 01-Jan-2022 row dropped, the
    # 31-Dec-2017 overlap deduped to the later (2018-2022) file's value.
    assert len(df) == 5
    by_date = {row["date"].isoformat(): row["audusd"] for _, row in df.iterrows()}
    assert by_date["2017-12-31"] == pytest.approx(0.9999)
    assert by_date["2024-01-04"] == pytest.approx(0.6800)


@responses.activate
def test_filters_to_requested_range(
    tmp_path: Path,
    hist_2014_2017_csv: bytes,
    hist_2018_2022_csv: bytes,
    current_csv: bytes,
) -> None:
    out = tmp_path / "audusd.parquet"
    _register_all_csv_sources(_all_sources(hist_2014_2017_csv, hist_2018_2022_csv, current_csv))

    audusd.fetch("2018-01-01", "2018-01-31", out)

    df = pd.read_parquet(out)
    assert len(df) == 1
    assert df["date"].iloc[0].isoformat() == "2018-01-02"


@responses.activate
def test_skips_fetch_when_cache_fresh(tmp_path: Path) -> None:
    out = tmp_path / "audusd.parquet"
    out.write_bytes(b"placeholder")

    # No mock registered — if `requests.get` is called, the test fails.
    audusd.fetch("2018-01-01", "2024-01-31", out, max_age_days=1.0)

    assert out.read_bytes() == b"placeholder"


@responses.activate
def test_force_bypasses_cache(
    tmp_path: Path,
    hist_2014_2017_csv: bytes,
    hist_2018_2022_csv: bytes,
    current_csv: bytes,
) -> None:
    out = tmp_path / "audusd.parquet"
    out.write_bytes(b"placeholder")

    _register_all_csv_sources(_all_sources(hist_2014_2017_csv, hist_2018_2022_csv, current_csv))

    audusd.fetch("2014-01-01", "2024-12-31", out, force=True)

    df = pd.read_parquet(out)
    assert "audusd" in df.columns


@responses.activate
def test_stale_cache_triggers_refetch(
    tmp_path: Path,
    hist_2014_2017_csv: bytes,
    hist_2018_2022_csv: bytes,
    current_csv: bytes,
) -> None:
    out = tmp_path / "audusd.parquet"
    out.write_bytes(b"placeholder")
    import os

    old = time.time() - 5 * 86400
    os.utime(out, (old, old))

    _register_all_csv_sources(_all_sources(hist_2014_2017_csv, hist_2018_2022_csv, current_csv))

    audusd.fetch("2014-01-01", "2024-12-31", out, max_age_days=1.0)

    df = pd.read_parquet(out)
    assert len(df) == 5


def test_parser_raises_when_series_missing() -> None:
    csv = _fake_rba_csv([("01-Jan-2010", "0.9", "0.5")]).replace(b"FXRUSD", b"FXRJPY")
    with pytest.raises(RuntimeError, match="FXRUSD"):
        audusd._parse_rba_table(csv, "fake.csv", audusd.SERIES_ID_AUDUSD)


@responses.activate
def test_retries_on_transient_failure(
    tmp_path: Path,
    hist_2014_2017_csv: bytes,
    hist_2018_2022_csv: bytes,
    current_csv: bytes,
) -> None:
    """tenacity should retry a 503 once before succeeding (waits no-op'd in conftest)."""
    out = tmp_path / "audusd.parquet"

    responses.add(responses.GET, audusd.URL_HIST_2014_2017, body=b"boom", status=503)
    responses.add(responses.GET, audusd.URL_HIST_2014_2017, body=hist_2014_2017_csv, status=200)
    responses.add(responses.GET, audusd.URL_HIST_2018_2022, body=hist_2018_2022_csv, status=200)
    responses.add(responses.GET, audusd.URL_CURRENT, body=current_csv, status=200)

    audusd.fetch("2014-01-01", "2024-12-31", out)

    df = pd.read_parquet(out)
    assert len(df) == 5
