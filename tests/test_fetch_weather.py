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


# ----------------------------- rate-limit classification -----------------------------


def test_classify_rate_limit_minutely() -> None:
    """A 'Minutely API request limit' reason → OpenMeteoMinutelyRateLimitError."""
    reason = "Minutely API request limit exceeded. Please try again in one minute."
    err = weather._classify_rate_limit(reason)
    assert isinstance(err, weather.OpenMeteoMinutelyRateLimitError)
    assert isinstance(err, weather.OpenMeteoRateLimitError)
    assert weather._is_retryable(err) is True  # short window, worth retrying


def test_classify_rate_limit_daily() -> None:
    """A 'Daily API request limit' reason → OpenMeteoDailyRateLimitError, NOT retryable."""
    reason = "Daily API request limit exceeded. Please try again tomorrow."
    err = weather._classify_rate_limit(reason)
    assert isinstance(err, weather.OpenMeteoDailyRateLimitError)
    assert weather._is_retryable(err) is False  # full day to wait — don't burn quota


def test_classify_rate_limit_unknown_is_conservative() -> None:
    """Unknown subtype (e.g. hourly) → base class, NOT retryable (safer)."""
    err = weather._classify_rate_limit("Some new throttle wording from a future API update")
    assert isinstance(err, weather.OpenMeteoRateLimitError)
    assert not isinstance(err, weather.OpenMeteoMinutelyRateLimitError)
    assert weather._is_retryable(err) is False


# ----------------------------- end-date clamping -----------------------------


def test_clamp_end_to_yesterday_clamps_today() -> None:
    today = dt.datetime.now(dt.UTC).date()
    yesterday = today - dt.timedelta(days=1)
    assert weather._clamp_end_to_yesterday(today.isoformat()) == yesterday.isoformat()


def test_clamp_end_to_yesterday_passes_through_old_dates() -> None:
    # Far in the past — passes through unchanged.
    assert weather._clamp_end_to_yesterday("2020-06-15") == "2020-06-15"


# ----------------------------- _cache_covers -----------------------------


def _valid_cache_frame(start: str, end: str) -> pd.DataFrame:
    """Build a synthetic cache parquet with date + at least one wx_* column.

    The stricter _cache_covers (spec §13.7 v2 resume hardening) requires
    both `date` AND at least one `wx_*` column to be present — defends
    against partial writes that landed the date column but lost the wx
    values mid-write.
    """
    dates = pd.date_range(start, end).date
    return pd.DataFrame({
        "date": dates,
        "wx_temp_max_c": [22.5] * len(dates),
    })


def test_cache_covers_true_when_range_fully_covered(tmp_path: Path) -> None:
    cache = tmp_path / "x.parquet"
    _valid_cache_frame("2024-01-01", "2024-01-31").to_parquet(
        cache, engine="pyarrow", compression="zstd", index=False,
    )
    assert weather._cache_covers(cache, "2024-01-05", "2024-01-25") is True


def test_cache_covers_false_when_range_partial(tmp_path: Path) -> None:
    cache = tmp_path / "x.parquet"
    _valid_cache_frame("2024-01-10", "2024-01-20").to_parquet(
        cache, engine="pyarrow", compression="zstd", index=False,
    )
    # Asks for a range that extends past the cache.
    assert weather._cache_covers(cache, "2024-01-05", "2024-01-25") is False


def test_cache_covers_false_when_file_missing(tmp_path: Path) -> None:
    assert weather._cache_covers(tmp_path / "nope.parquet", "2024-01-01", "2024-01-31") is False


def test_cache_covers_false_when_no_wx_columns(tmp_path: Path) -> None:
    """Defends against partial writes — date column only, wx columns missing."""
    cache = tmp_path / "bad.parquet"
    # Date column only, no wx_* — simulates a write that crashed mid-schema.
    pd.DataFrame({"date": pd.date_range("2024-01-01", "2024-01-31").date}).to_parquet(
        cache, engine="pyarrow", compression="zstd", index=False,
    )
    assert weather._cache_covers(cache, "2024-01-05", "2024-01-25") is False


def test_cache_covers_false_when_corrupt(tmp_path: Path) -> None:
    """Defends against truncated / non-parquet files (partial atomic writes)."""
    cache = tmp_path / "corrupt.parquet"
    cache.write_bytes(b"not a parquet file")
    assert weather._cache_covers(cache, "2024-01-05", "2024-01-25") is False


def test_cache_covers_forecast_only_accepts_post_2017_floor(tmp_path: Path) -> None:
    """forecast_only=True: a cache that starts at 2017-01-01 is valid even
    when the caller requested data back to 2016-09-01 — the pre-boundary
    range is legitimately absent in forecast-only mode."""
    cache = tmp_path / "fo.parquet"
    _valid_cache_frame("2017-01-01", "2026-04-30").to_parquet(
        cache, engine="pyarrow", compression="zstd", index=False,
    )
    # Hybrid-mode check: insufficient (would want 2016 data)
    assert weather._cache_covers(cache, "2016-09-01", "2026-04-30") is False
    # Forecast-only-mode check: sufficient (boundary is 2017-01-01)
    assert weather._cache_covers(
        cache, "2016-09-01", "2026-04-30", forecast_only=True
    ) is True


# ----------------------------- atomic write -----------------------------


@responses.activate
def test_fetch_one_atomic_write_no_tmp_remains(tmp_path: Path) -> None:
    """fetch_one writes to a .tmp then renames — no .tmp should survive."""
    responses.add(
        responses.GET, weather.FORECAST_URL,
        json=_open_meteo_response(["2024-01-01", "2024-01-02"]), status=200,
    )
    out_dir = tmp_path / "wx"
    weather.fetch_one(
        station_id="s1", lat=-33.93, lon=151.20,
        start="2024-01-01", end="2024-01-02",
        out_dir=out_dir, forecast_only=True,
    )
    parquets = list(out_dir.glob("*.parquet"))
    tmps = list(out_dir.glob("*.tmp"))
    assert len(parquets) == 1, f"expected one parquet, got {parquets}"
    assert tmps == [], f"unexpected .tmp leftover: {tmps}"


# ----------------------------- resume from partial cache -----------------------------


@responses.activate
def test_fetch_skips_cached_stations(tmp_path: Path) -> None:
    """A station with a valid cache covering the requested range is NOT re-fetched.

    Simulates the recovery-from-interruption pattern: a previous run
    completed N of M stations; the next run picks up where it stopped.
    """
    out_dir = tmp_path / "wx"
    out_dir.mkdir()
    # Pre-populate s1's cache (covers the requested range).
    _valid_cache_frame("2024-01-01", "2024-01-31").to_parquet(
        out_dir / "s1.parquet", engine="pyarrow", compression="zstd", index=False,
    )
    # Only s2's forecast URL needs to mock — s1 should never hit the API.
    responses.add(
        responses.GET, weather.FORECAST_URL,
        json=_open_meteo_response(["2024-01-01", "2024-01-31"]), status=200,
    )

    stations = _stations_parquet(tmp_path, [
        {"station_id": "s1", "lat": -33.93, "lon": 151.20},
        {"station_id": "s2", "lat": -32.93, "lon": 151.78},
    ])
    weather.fetch(
        stations, "2024-01-01", "2024-01-31", out_dir,
        inter_call_seconds=0.0, forecast_only=True,
    )
    # Exactly one API call (s2 only); s1 was a cache hit.
    assert len(responses.calls) == 1
    # Both stations have parquets in the output.
    parquets = sorted(p.name for p in out_dir.glob("*.parquet"))
    assert parquets == ["s1.parquet", "s2.parquet"]


# ----------------------------- fetch (full path) -----------------------------


@responses.activate
def test_fetch_writes_per_station_parquet(tmp_path: Path, stations_two: Path) -> None:
    out_dir = tmp_path / "weather"
    # 2024-01-01 is post-WEATHER_FORECAST_COVERAGE_START, so routes to the
    # Historical Forecast API (spec §13.7).
    responses.add(
        responses.GET,
        weather.FORECAST_URL,
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

    # 2024-01-01 routes to the forecast API under v2.0 hybrid logic.
    responses.add(
        responses.GET,
        weather.FORECAST_URL,
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
    # Post-2017 dates route to forecast (spec §13.7).
    responses.add(
        responses.GET,
        weather.FORECAST_URL,
        json=_open_meteo_response(["2024-01-01"]),
        status=200,
    )

    weather.fetch(stations, "2024-01-01", "2024-01-01", out_dir, inter_call_seconds=0)

    files = sorted(p.name for p in out_dir.glob("*.parquet"))
    assert files == ["s1.parquet"]


@responses.activate
def test_fetch_continues_on_per_station_failure(tmp_path: Path, stations_two: Path) -> None:
    """One station failing shouldn't kill the whole run.

    Weather has a custom 7-attempt retry (vs the project-wide
    RETRY_MAX_ATTEMPTS=5) to span Open-Meteo's per-minute throttle
    window. Mock 7 sequential 500s for s1 (exhausts the retry budget),
    then a success for s2.
    """
    out_dir = tmp_path / "weather"
    # First station call: HTTP 500 (exhausts the 7-attempt retry budget).
    for _ in range(7):
        responses.add(responses.GET, weather.FORECAST_URL, status=500)
    # Second station call: success. Post-2017 routes to forecast (spec §13.7).
    responses.add(
        responses.GET,
        weather.FORECAST_URL,
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

    # Today is well after 2017-01-01, so the call routes to the forecast API.
    responses.add(
        responses.GET,
        weather.FORECAST_URL,
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
        weather.FORECAST_URL,
        json=_open_meteo_response(["2024-01-01"]),
        status=200,
    )

    weather.fetch(stations_two, "2024-01-01", "2024-01-01", out_dir, inter_call_seconds=0)

    last = responses.calls[-1].request
    assert "timezone=Australia%2FSydney" in last.url or "timezone=Australia/Sydney" in last.url


@responses.activate
def test_fetch_one_returns_path_on_success(tmp_path: Path) -> None:
    out_dir = tmp_path / "weather"
    # Post-2017 → forecast API under v2.0 hybrid logic.
    responses.add(
        responses.GET,
        weather.FORECAST_URL,
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


# ----------------------------- hybrid routing (spec §13.7) -----------------------------


@responses.activate
def test_fetch_one_uses_archive_for_pre_2017_dates(tmp_path: Path) -> None:
    """Dates entirely before WEATHER_FORECAST_COVERAGE_START hit the ERA5 archive only."""
    out_dir = tmp_path / "weather"
    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json=_open_meteo_response(["2016-09-01", "2016-09-02"]),
        status=200,
    )

    weather.fetch_one("s1", -33.93, 151.20, "2016-09-01", "2016-09-02", out_dir)

    archive_calls = [c for c in responses.calls if c.request.url.startswith(weather.ARCHIVE_URL)]
    forecast_calls = [c for c in responses.calls if c.request.url.startswith(weather.FORECAST_URL)]
    assert len(archive_calls) == 1
    assert len(forecast_calls) == 0


@responses.activate
def test_fetch_one_uses_forecast_for_post_2017_dates(tmp_path: Path) -> None:
    """Dates entirely on/after WEATHER_FORECAST_COVERAGE_START hit the forecast API only."""
    out_dir = tmp_path / "weather"
    responses.add(
        responses.GET,
        weather.FORECAST_URL,
        json=_open_meteo_response(["2024-01-01", "2024-01-02"]),
        status=200,
    )

    weather.fetch_one("s1", -33.93, 151.20, "2024-01-01", "2024-01-02", out_dir)

    archive_calls = [c for c in responses.calls if c.request.url.startswith(weather.ARCHIVE_URL)]
    forecast_calls = [c for c in responses.calls if c.request.url.startswith(weather.FORECAST_URL)]
    assert len(forecast_calls) == 1
    assert len(archive_calls) == 0


@responses.activate
def test_fetch_one_straddle_calls_both(tmp_path: Path) -> None:
    """A range that straddles 2017-01-01 hits archive for the pre-2017 half
    and forecast for the post-2017 half, deduplicating at the seam."""
    out_dir = tmp_path / "weather"

    archive_dates = [d.isoformat() for d in pd.date_range("2016-12-15", "2016-12-31").date]
    forecast_dates = [d.isoformat() for d in pd.date_range("2017-01-01", "2017-01-15").date]

    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json=_open_meteo_response(archive_dates),
        status=200,
    )
    responses.add(
        responses.GET,
        weather.FORECAST_URL,
        json=_open_meteo_response(forecast_dates),
        status=200,
    )

    weather.fetch_one("s1", -33.93, 151.20, "2016-12-15", "2017-01-15", out_dir)

    archive_calls = [c for c in responses.calls if c.request.url.startswith(weather.ARCHIVE_URL)]
    forecast_calls = [c for c in responses.calls if c.request.url.startswith(weather.FORECAST_URL)]
    assert len(archive_calls) == 1, f"expected 1 archive call, got {len(archive_calls)}"
    assert len(forecast_calls) == 1, f"expected 1 forecast call, got {len(forecast_calls)}"

    df = pd.read_parquet(out_dir / "s1.parquet")
    # 17 archive + 15 forecast = 32 unique dates, no duplicates at the seam.
    assert len(df) == 32
    assert df["date"].is_unique
    assert df["date"].min() == dt.date(2016, 12, 15)
    assert df["date"].max() == dt.date(2017, 1, 15)


@responses.activate
def test_fetch_one_safety_net_backfills_all_null_forecast_rows(tmp_path: Path) -> None:
    """Per preflight: forecast API returns null precip on 2017-01-01.

    If a forecast-window row has every wx_* null, a follow-up archive
    backfill must populate it. This test makes the entire 2017-01-01 row
    null (extreme case) and confirms the archive is hit a second time
    for that date and the values land in the cached parquet.
    """
    out_dir = tmp_path / "weather"
    forecast_dates = ["2017-01-01", "2017-01-02"]

    # 2017-01-01 returns all nulls on the forecast call (worst-case version
    # of the preflight's precip-null finding).
    forecast_payload = _open_meteo_response(
        forecast_dates,
        temp_max=[None, 25.0],  # type: ignore[list-item]
        temp_min=[None, 14.0],  # type: ignore[list-item]
        precip=[None, 0.5],  # type: ignore[list-item]
        wind=[None, 12.0],  # type: ignore[list-item]
        code=[None, 1],  # type: ignore[list-item]
    )
    responses.add(responses.GET, weather.FORECAST_URL, json=forecast_payload, status=200)
    # The backfill archive call should ask for 2017-01-01 specifically.
    responses.add(
        responses.GET,
        weather.ARCHIVE_URL,
        json=_open_meteo_response(
            ["2017-01-01"], temp_max=[22.0], temp_min=[13.0], precip=[0.0], wind=[10.0], code=[0]
        ),
        status=200,
    )

    weather.fetch_one("s1", -33.93, 151.20, "2017-01-01", "2017-01-02", out_dir)

    archive_calls = [c for c in responses.calls if c.request.url.startswith(weather.ARCHIVE_URL)]
    forecast_calls = [c for c in responses.calls if c.request.url.startswith(weather.FORECAST_URL)]
    assert len(forecast_calls) == 1
    assert len(archive_calls) == 1  # the backfill

    df = pd.read_parquet(out_dir / "s1.parquet").sort_values("date").reset_index(drop=True)
    assert len(df) == 2
    # 2017-01-01 row now carries archive values, not nulls.
    assert float(df.loc[df["date"] == dt.date(2017, 1, 1), "wx_temp_max_c"].iloc[0]) == 22.0
    # 2017-01-02 keeps its forecast values.
    assert float(df.loc[df["date"] == dt.date(2017, 1, 2), "wx_temp_max_c"].iloc[0]) == 25.0


def test_split_at_boundary_pre_only() -> None:
    archive, forecast = weather._split_at_boundary("2016-09-01", "2016-12-31", "2017-01-01")
    assert archive == ("2016-09-01", "2016-12-31")
    assert forecast is None


def test_split_at_boundary_post_only() -> None:
    archive, forecast = weather._split_at_boundary("2017-01-01", "2017-12-31", "2017-01-01")
    assert archive is None
    assert forecast == ("2017-01-01", "2017-12-31")


def test_split_at_boundary_straddles() -> None:
    archive, forecast = weather._split_at_boundary("2016-12-15", "2017-01-15", "2017-01-01")
    assert archive == ("2016-12-15", "2016-12-31")
    assert forecast == ("2017-01-01", "2017-01-15")
