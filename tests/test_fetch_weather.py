"""Hermetic tests for fetch.weather. Open-Meteo archive API mocked with `responses`."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import responses

from fuel_pred.fetch import weather


def _open_meteo_response(
    dates: list[str],
    *,
    temp_max: list[float] | None = None,
    temp_min: list[float] | None = None,
    precip: list[float] | None = None,
    wind: list[float] | None = None,
    code: list[int] | None = None,
) -> dict[str, object]:
    """Build a synthetic Open-Meteo daily response."""
    n = len(dates)
    return {
        "latitude": -33.93,
        "longitude": 151.20,
        "timezone": weather.TIMEZONE,
        "daily": {
            "time": dates,
            "temperature_2m_max": temp_max if temp_max is not None else [22.5] * n,
            "temperature_2m_min": temp_min if temp_min is not None else [12.0] * n,
            "precipitation_sum": precip if precip is not None else [0.0] * n,
            "wind_speed_10m_max": wind if wind is not None else [15.0] * n,
            "weather_code": code if code is not None else [0] * n,
        },
    }


def _stations_parquet(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "stations.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    return p


@pytest.fixture
def stations_two(tmp_path: Path) -> Path:
    return _stations_parquet(
        tmp_path,
        [
            {"station_id": "s1", "lat": -33.93, "lon": 151.20},
            {"station_id": "s2", "lat": -32.93, "lon": 151.78},
        ],
    )


# ----------------------------- _frame_from_payload -----------------------------


def test_frame_from_payload_produces_expected_schema() -> None:
    payload = _open_meteo_response(["2024-01-01", "2024-01-02"])
    df = weather._frame_from_payload(payload)
    assert list(df.columns) == list(weather.OUTPUT_COLUMNS)
    assert len(df) == 2
    # Date column must be plain `date`, not datetime, so it joins cleanly
    # with the FuelCheck date column.
    assert isinstance(df["date"].iloc[0], dt.date)
    assert not isinstance(df["date"].iloc[0], dt.datetime)


def test_frame_from_payload_typing() -> None:
    """Numeric columns must be Float64/Int64 so the parquet schema is stable
    even if a station happens to return only nulls (e.g. ocean-located row)."""
    payload = _open_meteo_response(
        ["2024-01-01"],
        temp_max=[None],  # type: ignore[list-item]
        code=[None],  # type: ignore[list-item]
    )
    df = weather._frame_from_payload(payload)
    assert df["wx_temp_max_c"].dtype.name == "Float64"
    assert df["wx_weather_code"].dtype.name == "Int64"


def test_frame_from_payload_raises_on_missing_variable() -> None:
    payload = {"daily": {"time": ["2024-01-01"], "temperature_2m_max": [22.0]}}
    with pytest.raises(RuntimeError, match="missing variable"):
        weather._frame_from_payload(payload)


def test_frame_from_payload_raises_on_missing_daily_block() -> None:
    with pytest.raises(RuntimeError, match="payload shape"):
        weather._frame_from_payload({"latitude": -33.0})


# ----------------------------- end-date clamping -----------------------------


def test_clamp_end_to_yesterday_clamps_today() -> None:
    today = dt.datetime.now(dt.UTC).date()
    yesterday = today - dt.timedelta(days=1)
    assert weather._clamp_end_to_yesterday(today.isoformat()) == yesterday.isoformat()


def test_clamp_end_to_yesterday_passes_through_old_dates() -> None:
    # Far in the past — passes through unchanged.
    assert weather._clamp_end_to_yesterday("2020-06-15") == "2020-06-15"


# ----------------------------- _cache_covers -----------------------------


def test_cache_covers_true_when_range_fully_covered(tmp_path: Path) -> None:
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", "2024-01-31").date})
    cache = tmp_path / "x.parquet"
    df.to_parquet(cache, engine="pyarrow", compression="zstd", index=False)
    assert weather._cache_covers(cache, "2024-01-05", "2024-01-25") is True


def test_cache_covers_false_when_range_partial(tmp_path: Path) -> None:
    df = pd.DataFrame({"date": pd.date_range("2024-01-10", "2024-01-20").date})
    cache = tmp_path / "x.parquet"
    df.to_parquet(cache, engine="pyarrow", compression="zstd", index=False)
    # Asks for a range that extends past the cache.
    assert weather._cache_covers(cache, "2024-01-05", "2024-01-25") is False


def test_cache_covers_false_when_file_missing(tmp_path: Path) -> None:
    assert weather._cache_covers(tmp_path / "nope.parquet", "2024-01-01", "2024-01-31") is False


# ----------------------------- fetch (full path) -----------------------------


@responses.activate
def test_fetch_writes_per_station_parquet(tmp_path: Path, stations_two: Path) -> None:
    out_dir = tmp_path / "weather"
    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json=_open_meteo_response(["2024-01-01", "2024-01-02"]),
        status=200,
    )

    weather.fetch(stations_two, "2024-01-01", "2024-01-02", out_dir, inter_call_seconds=0)

    files = sorted(p.name for p in out_dir.glob("*.parquet"))
    assert files == ["s1.parquet", "s2.parquet"]
    df = pd.read_parquet(out_dir / "s1.parquet")
    assert list(df.columns) == list(weather.OUTPUT_COLUMNS)
    assert len(df) == 2


@responses.activate
def test_fetch_skips_when_cache_covers(tmp_path: Path, stations_two: Path) -> None:
    """Existing parquet whose date range covers the request: don't refetch."""
    out_dir = tmp_path / "weather"
    out_dir.mkdir()
    cached = pd.DataFrame(
        {
            "date": pd.date_range("2023-12-01", "2024-12-31").date,
            "wx_temp_max_c": [22.0] * 397,
            "wx_temp_min_c": [12.0] * 397,
            "wx_precipitation_mm": [0.0] * 397,
            "wx_wind_speed_max_kmh": [15.0] * 397,
            "wx_weather_code": [0] * 397,
        }
    )
    cached.to_parquet(out_dir / "s1.parquet", engine="pyarrow", compression="zstd", index=False)
    cached.to_parquet(out_dir / "s2.parquet", engine="pyarrow", compression="zstd", index=False)

    # No mock registered — if the fetcher hits the network, the test fails.
    weather.fetch(stations_two, "2024-01-01", "2024-01-31", out_dir, inter_call_seconds=0)


@responses.activate
def test_fetch_force_bypasses_cache(tmp_path: Path, stations_two: Path) -> None:
    out_dir = tmp_path / "weather"
    out_dir.mkdir()
    # Drop a placeholder so we know force overwrote it.
    (out_dir / "s1.parquet").write_bytes(b"placeholder")
    (out_dir / "s2.parquet").write_bytes(b"placeholder")

    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json=_open_meteo_response(["2024-01-01"]),
        status=200,
    )
    weather.fetch(
        stations_two, "2024-01-01", "2024-01-01", out_dir, force=True, inter_call_seconds=0
    )

    df = pd.read_parquet(out_dir / "s1.parquet")
    assert len(df) == 1


@responses.activate
def test_fetch_handles_missing_lat_lon(tmp_path: Path) -> None:
    """Stations with NaN lat/lon are skipped with a warning, not a failure."""
    stations = _stations_parquet(
        tmp_path,
        [
            {"station_id": "s1", "lat": -33.93, "lon": 151.20},
            {"station_id": "s2", "lat": None, "lon": None},
        ],
    )
    out_dir = tmp_path / "weather"
    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json=_open_meteo_response(["2024-01-01"]),
        status=200,
    )

    weather.fetch(stations, "2024-01-01", "2024-01-01", out_dir, inter_call_seconds=0)

    files = sorted(p.name for p in out_dir.glob("*.parquet"))
    assert files == ["s1.parquet"]


@responses.activate
def test_fetch_continues_on_per_station_failure(tmp_path: Path, stations_two: Path) -> None:
    """One station failing shouldn't kill the whole run."""
    out_dir = tmp_path / "weather"
    # First station call: HTTP 500 (after 5 retries → permanent failure).
    # Second station call: success.
    for _ in range(weather.config.RETRY_MAX_ATTEMPTS):
        responses.add(responses.GET, weather.ARCHIVE_URL, status=500)
    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json=_open_meteo_response(["2024-01-01"]),
        status=200,
    )

    weather.fetch(stations_two, "2024-01-01", "2024-01-01", out_dir, inter_call_seconds=0)

    files = sorted(p.name for p in out_dir.glob("*.parquet"))
    # Only s2 succeeded; s1 logged a failure.
    assert files == ["s2.parquet"]


@responses.activate
def test_fetch_clamps_end_date_when_in_future(tmp_path: Path, stations_two: Path) -> None:
    """end='today' must be silently clamped to yesterday."""
    out_dir = tmp_path / "weather"
    today = dt.datetime.now(dt.UTC).date()
    yesterday = today - dt.timedelta(days=1)

    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json=_open_meteo_response([yesterday.isoformat()]),
        status=200,
    )

    weather.fetch(
        stations_two, "2024-01-01", today.isoformat(), out_dir, inter_call_seconds=0
    )

    # Verify the actual request used yesterday's date.
    last = responses.calls[-1].request
    assert f"end_date={yesterday.isoformat()}" in last.url


@responses.activate
def test_fetch_uses_sydney_timezone(tmp_path: Path, stations_two: Path) -> None:
    """Day boundaries must be in Australia/Sydney so weather joins to FuelCheck dates."""
    out_dir = tmp_path / "weather"
    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json=_open_meteo_response(["2024-01-01"]),
        status=200,
    )

    weather.fetch(stations_two, "2024-01-01", "2024-01-01", out_dir, inter_call_seconds=0)

    last = responses.calls[-1].request
    assert "timezone=Australia%2FSydney" in last.url or "timezone=Australia/Sydney" in last.url


@responses.activate
def test_fetch_one_returns_path_on_success(tmp_path: Path) -> None:
    out_dir = tmp_path / "weather"
    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json=_open_meteo_response(["2024-01-01"]),
        status=200,
    )
    path = weather.fetch_one("s1", -33.93, 151.20, "2024-01-01", "2024-01-01", out_dir)
    assert path is not None
    assert path.exists()


@responses.activate
def test_open_meteo_error_payload_raises() -> None:
    """If Open-Meteo returns a JSON payload with `error: true`, raise loudly."""
    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json={"error": True, "reason": "Latitude must be in range -90..90"},
        status=400,
    )
    with pytest.raises(Exception):  # noqa: B017
        weather._request_daily(999.0, 999.0, "2024-01-01", "2024-01-01")
