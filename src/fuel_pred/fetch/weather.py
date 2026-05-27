"""Fetch daily weather aggregates from Open-Meteo for each station lat/lon.

Hybrid source — v2.0 leakage fix (spec §13.7):

- **Historical Forecast API** (`historical-forecast-api.open-meteo.com/v1/forecast`)
  for dates ``>= WEATHER_FORECAST_COVERAGE_START`` (2017-01-01).
  This returns the actual NWP day-ahead forecast as it was operationally
  issued — what a deployed predictor would have had — instead of ERA5
  reanalysis truth. Joining these values onto a panel row at date ``t``
  with the 1-day shift in ``add_weather_features`` then gives the model
  the forecast for ``t+1`` as known on ``t``.

- **Historical Weather (ERA5) Archive** (`archive-api.open-meteo.com/v1/archive`)
  as a fallback for dates before the forecast API's Australian coverage
  begins. The pre-2017 window (2016-09 → 2016-12) sits entirely inside
  the training fold, so val/test metrics are unaffected. Documented as
  a known persistence-proxy in the spec.

Per the preflight (`docs/research/2026-05_weather_leakage_preflight.md`),
the boundary day 2017-01-01 has a one-day precipitation gap on the
forecast API (other four variables populate). Any rows where all five
``wx_*`` variables are null after the forecast call are backfilled from
a follow-up archive call so the cached parquet has no all-null rows from
fixable causes.

Granularity: daily, with day boundaries in `Australia/Sydney` local time
so that the resulting `date` column joins cleanly to the FuelCheck-derived
`date` column in `fuel_daily.parquet` (also a local-date).

Per-station caching: `data/raw/weather/<station_id>.parquet`. Stations
share lat/lon with thousands of neighbours so this isn't optimal — a
location-keyed cache would be tighter — but per-station keeps the cache
detection trivial (file exists + covers requested range = hit) and
matches the spec layout.

Variables returned (spec.md §7.6):

    wx_temp_max_c           # Open-Meteo: temperature_2m_max (°C)
    wx_temp_min_c           # Open-Meteo: temperature_2m_min (°C)
    wx_precipitation_mm     # Open-Meteo: precipitation_sum (mm)
    wx_wind_speed_max_kmh   # Open-Meteo: wind_speed_10m_max (km/h)
    wx_weather_code         # Open-Meteo: weather_code (WMO code)

Spec: spec.md §5.1, §7.6, §13.7.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from fuel_pred import config

logger = logging.getLogger(__name__)


class OpenMeteoRateLimitError(RuntimeError):
    """Raised on HTTP 429 from Open-Meteo.

    Each 429 still counts against the per-minute / per-hour / per-day
    quotas, so retrying multiplies the burn. The subclasses below carry
    the severity so tenacity / the orchestrator can react appropriately:

    - Minutely (600/min) — short-lived; safe to retry after ~65s.
    - Hourly  (5000/hr) — long-lived; not retried in-process; abort.
    - Daily   (10000/day) — abort, resume tomorrow.
    - Unknown — treat as Hourly (conservative: abort).
    """


class OpenMeteoMinutelyRateLimitError(OpenMeteoRateLimitError):
    """600 calls/min burst limit. Retryable after ~65s wait."""


class OpenMeteoDailyRateLimitError(OpenMeteoRateLimitError):
    """10000 calls/day cap. NOT retryable — wait for UTC midnight reset."""


def _classify_rate_limit(reason: str) -> OpenMeteoRateLimitError:
    """Pick the right subclass based on the API's `reason` string."""
    low = reason.lower()
    if "minute" in low:
        return OpenMeteoMinutelyRateLimitError(reason)
    if "daily" in low or "day" in low:
        return OpenMeteoDailyRateLimitError(reason)
    # Unknown subtype (hourly, or new wording) — be conservative, surface
    # as the base class which the orchestrator treats as abort-worthy.
    return OpenMeteoRateLimitError(reason)


def _is_retryable(exc: BaseException) -> bool:
    """Tenacity predicate: retry transient errors + minutely 429s only.

    Daily 429s (and unclassified 429s) abort immediately — retrying
    burns more quota for nothing. Minutely 429s wait ~65s via the
    tenacity exponential backoff and try again — the per-minute window
    is short enough that retry pays off.
    """
    if isinstance(exc, OpenMeteoMinutelyRateLimitError):
        return True  # short-lived throttle, retry pays off
    if isinstance(exc, OpenMeteoRateLimitError):
        return False  # daily / unknown — abort, don't burn quota
    # Other RequestException / HTTPError / ConnectionError / Timeout: retry.
    return isinstance(exc, requests.RequestException | requests.HTTPError | OSError)

ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Local time matters — Open-Meteo aggregates daily values in this timezone,
# so the `date` column in the cached parquet is a local-date that joins
# cleanly against the FuelCheck-derived date in `fuel_daily.parquet`.
TIMEZONE: str = "Australia/Sydney"

# Open-Meteo daily variable names → our spec'd column names.
DAILY_VARIABLES: dict[str, str] = {
    "temperature_2m_max": "wx_temp_max_c",
    "temperature_2m_min": "wx_temp_min_c",
    "precipitation_sum": "wx_precipitation_mm",
    "wind_speed_10m_max": "wx_wind_speed_max_kmh",
    "weather_code": "wx_weather_code",
}

OUTPUT_COLUMNS: tuple[str, ...] = ("date", *DAILY_VARIABLES.values())
WX_VALUE_COLUMNS: tuple[str, ...] = tuple(DAILY_VARIABLES.values())

# Polite delay between station calls. Open-Meteo's free tier allows
# ~600 calls/min; 0.1s = 600/min upper bound. We're well under.
DEFAULT_INTER_CALL_SECONDS: float = 0.1


@retry(
    # Weather is the only fetcher hit by Open-Meteo's per-minute throttle;
    # 7 attempts × exp backoff (max 90s) gives ~2+4+8+16+32+64 = 126s of
    # total wait — covers a full 60s window plus headroom for retry-on-retry.
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=config.RETRY_BACKOFF_SECONDS, max=90),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _request_daily(lat: float, lon: float, start: str, end: str) -> dict[str, Any]:
    """One Open-Meteo *archive* (ERA5) call. Retries on transient errors.

    HTTP 429 (rate limit) is NOT retried — surfaced as
    OpenMeteoRateLimitError so the caller can apply a circuit breaker
    instead of multiplying the quota burn via tenacity retries.
    """
    return _open_meteo_get(ARCHIVE_URL, lat, lon, start, end)


@retry(
    # Weather is the only fetcher hit by Open-Meteo's per-minute throttle;
    # 7 attempts × exp backoff (max 90s) gives ~2+4+8+16+32+64 = 126s of
    # total wait — covers a full 60s window plus headroom for retry-on-retry.
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=config.RETRY_BACKOFF_SECONDS, max=90),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _request_daily_forecast(
    lat: float, lon: float, start: str, end: str
) -> dict[str, Any]:
    """One Open-Meteo *Historical Forecast* call. Retries on transient errors.

    Identical call shape to :func:`_request_daily` but hits the forecast
    endpoint (`historical-forecast-api.open-meteo.com`) instead of the
    ERA5 archive. The returned payload schema is the same.

    HTTP 429 surfaces immediately as OpenMeteoRateLimitError (no retry).
    """
    return _open_meteo_get(FORECAST_URL, lat, lon, start, end)


def _open_meteo_get(
    url: str, lat: float, lon: float, start: str, end: str
) -> dict[str, Any]:
    """Shared HTTP shim for the two Open-Meteo endpoints (same params shape)."""
    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARIABLES.keys()),
        "timezone": TIMEZONE,
    }
    # If a key is configured in .env, send it (raises Open-Meteo rate limits
    # ~10x). Free tier is keyless and works fine for forecast-only mode.
    if config.OPENMETEO_API_KEY:
        params["apikey"] = config.OPENMETEO_API_KEY
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"},
        timeout=config.REQUEST_TIMEOUT,
    )
    # 429 → classified by reason. Minutely → tenacity retries with wait.
    # Daily → abort. See _classify_rate_limit and _is_retryable.
    if response.status_code == 429:
        reason = ""
        try:
            reason = response.json().get("reason", "")
        except Exception:
            reason = response.text[:200]
        raise _classify_rate_limit(
            f"HTTP 429 from Open-Meteo ({url}): {reason}"
        )
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    if "error" in body and body.get("error"):
        raise RuntimeError(
            f"Open-Meteo error for ({lat}, {lon}) at {url}: {body.get('reason', body)}"
        )
    return body


def _frame_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert an Open-Meteo `daily` JSON block into our spec'd DataFrame."""
    daily = payload.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise RuntimeError(f"unexpected Open-Meteo payload shape: keys={list(payload.keys())}")

    df = pd.DataFrame({"date": pd.to_datetime(daily["time"]).date})
    for src, dst in DAILY_VARIABLES.items():
        if src not in daily:
            raise RuntimeError(f"missing variable {src!r} in Open-Meteo payload")
        df[dst] = daily[src]

    # Force types so the parquet schema is stable across stations.
    df["wx_temp_max_c"] = pd.to_numeric(df["wx_temp_max_c"], errors="coerce").astype("Float64")
    df["wx_temp_min_c"] = pd.to_numeric(df["wx_temp_min_c"], errors="coerce").astype("Float64")
    df["wx_precipitation_mm"] = pd.to_numeric(df["wx_precipitation_mm"], errors="coerce").astype(
        "Float64"
    )
    df["wx_wind_speed_max_kmh"] = pd.to_numeric(
        df["wx_wind_speed_max_kmh"], errors="coerce"
    ).astype("Float64")
    df["wx_weather_code"] = pd.to_numeric(df["wx_weather_code"], errors="coerce").astype("Int64")

    return df.loc[:, list(OUTPUT_COLUMNS)]


def _cache_covers(
    path: Path, start: str, end: str, *, forecast_only: bool = False
) -> bool:
    """Return True if the cached parquet covers the full requested range.

    Validates that:
      1. the file exists and is readable (not a half-written .tmp leftover
         or a corrupt parquet from a previous crash);
      2. the schema includes `date` and at least one `wx_*` value column —
         defends against the rare case of a parquet that wrote its date
         column but failed mid-write on the wx values;
      3. the cached date range covers the requested [start, end] window.
         In forecast-only mode the effective start is the later of the
         requested start and the forecast coverage boundary, since the
         pre-boundary range is legitimately absent.

    A False return triggers a re-fetch — the cost is one API call, so we
    err toward False whenever there's any ambiguity.
    """
    if not path.exists():
        return False
    try:
        # Read the whole thing once (small per-station files, ~35KB) so we
        # can verify both date AND wx columns are present and readable.
        cached = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("could not read cache %s (%s) — re-fetching", path, exc)
        return False
    if cached.empty or "date" not in cached.columns:
        return False
    if not any(c in cached.columns for c in WX_VALUE_COLUMNS):
        logger.warning(
            "cache %s has date column but no wx_* values — re-fetching", path,
        )
        return False
    cached_min = pd.to_datetime(cached["date"].min()).date()
    cached_max = pd.to_datetime(cached["date"].max()).date()
    requested_start = dt.date.fromisoformat(start)
    requested_end = dt.date.fromisoformat(end)
    if forecast_only:
        boundary = dt.date.fromisoformat(config.WEATHER_FORECAST_COVERAGE_START)
        effective_start = max(requested_start, boundary)
    else:
        effective_start = requested_start
    return bool(cached_min <= effective_start and cached_max >= requested_end)


def _clamp_end_to_yesterday(end: str) -> str:
    """ERA5 has a ~5-day lag. Asking for today/tomorrow returns nulls or 400s.

    The forecast API has a similar publication boundary (it stitches each run's
    initial hours into a continuous timeseries; very recent dates haven't been
    processed yet). Clamping to yesterday is conservative for both endpoints.

    We clamp `end` to yesterday (in UTC; the timezone offset is small enough
    that one extra day of buffer is fine) so callers don't have to do their
    own date arithmetic. Logged as INFO when clamping happens.
    """
    requested = dt.date.fromisoformat(end)
    yesterday = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1)
    if requested > yesterday:
        logger.info(
            "clamping end %s -> %s (Open-Meteo publication lag)",
            end, yesterday.isoformat(),
        )
        return yesterday.isoformat()
    return end


def _split_at_boundary(
    start: str, end: str, boundary: str
) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """Split `[start, end]` at `boundary` (forecast-API coverage start).

    Returns ``(archive_range, forecast_range)``; either may be ``None``.

    - ``archive_range = (start, min(end, boundary-1))`` when ``start < boundary``.
    - ``forecast_range = (max(start, boundary), end)`` when ``end >= boundary``.
    """
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    boundary_d = dt.date.fromisoformat(boundary)

    archive_range: tuple[str, str] | None = None
    forecast_range: tuple[str, str] | None = None

    if start_d < boundary_d:
        archive_end = min(end_d, boundary_d - dt.timedelta(days=1))
        if archive_end >= start_d:
            archive_range = (start_d.isoformat(), archive_end.isoformat())

    if end_d >= boundary_d:
        forecast_start = max(start_d, boundary_d)
        forecast_range = (forecast_start.isoformat(), end_d.isoformat())

    return archive_range, forecast_range


def _all_wx_null_dates(df: pd.DataFrame) -> list[dt.date]:
    """Return dates from `df` where every wx_* value is null.

    These are candidates for a follow-up archive backfill — the forecast
    API has known one-day gaps at its coverage boundary.
    """
    if df.empty:
        return []
    cols = [c for c in WX_VALUE_COLUMNS if c in df.columns]
    if not cols:
        return []
    all_null = df[cols].isna().all(axis=1)
    return [d for d in df.loc[all_null, "date"].tolist()]


def _fetch_hybrid(
    lat: float, lon: float, start: str, end: str, *, forecast_only: bool = False
) -> pd.DataFrame:
    """Build a per-station weather frame for `[start, end]` using the hybrid
    archive/forecast strategy.

    Pipeline (matches `docs/research/2026-05_weather_leakage_fix.md`
    §"Pipeline changes / fetch/weather.py"):

    1. Split the requested range at ``WEATHER_FORECAST_COVERAGE_START``.
    2. Fetch the archive portion (if any) and the forecast portion (if any).
    3. Concatenate, drop duplicate ``date`` rows (later — forecast — wins).
    4. **Safety net:** any rows where all five ``wx_*`` are null get
       backfilled by a follow-up archive call for those specific dates.
       Handles the known precip-null on 2017-01-01 and any analogous
       single-day gaps elsewhere in the forecast API.

    Args:
        forecast_only: if True, the archive (ERA5) portion is skipped
            entirely — only the forecast API is hit. Dates before
            ``WEATHER_FORECAST_COVERAGE_START`` are dropped from the
            output frame (the panel join in ``make_features`` will see
            null wx values for those dates). Useful when the archive API
            is rate-limited or unavailable.
    """
    archive_range, forecast_range = _split_at_boundary(
        start, end, config.WEATHER_FORECAST_COVERAGE_START
    )
    if forecast_only:
        archive_range = None  # explicitly drop pre-2017 fallback

    pieces: list[pd.DataFrame] = []
    if archive_range is not None:
        a_start, a_end = archive_range
        logger.info(
            "weather fetch: archive (ERA5) for (%s, %s) %s..%s",
            lat, lon, a_start, a_end,
        )
        pieces.append(_frame_from_payload(_request_daily(lat, lon, a_start, a_end)))
    if forecast_range is not None:
        f_start, f_end = forecast_range
        logger.info(
            "weather fetch: forecast (Historical Forecast API) for (%s, %s) %s..%s",
            lat, lon, f_start, f_end,
        )
        pieces.append(
            _frame_from_payload(_request_daily_forecast(lat, lon, f_start, f_end))
        )

    if not pieces:
        # Defensive — would mean start > end. Return an empty frame with
        # the expected schema so callers don't crash on schema lookups.
        return _frame_from_payload({"daily": {"time": [], **{k: [] for k in DAILY_VARIABLES}}})

    df = pd.concat(pieces, ignore_index=True)
    # Forecast rows are appended last; keep them when a date appears in both.
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(
        drop=True
    )

    # Safety-net backfill from archive for any all-null rows in the forecast window.
    # Skipped in forecast_only mode (would defeat the point of avoiding archive).
    null_dates = _all_wx_null_dates(df)
    if null_dates and forecast_range is not None and not forecast_only:
        # Backfill is only meaningful on the forecast side (the archive call
        # produced concrete values where it covered). Reduce to forecast-window
        # dates and call the archive endpoint for each contiguous run.
        f_start = dt.date.fromisoformat(forecast_range[0])
        f_end = dt.date.fromisoformat(forecast_range[1])
        backfill_dates = sorted(d for d in null_dates if f_start <= d <= f_end)
        if backfill_dates:
            backfill_runs = _contiguous_runs(backfill_dates)
            for run_start, run_end in backfill_runs:
                logger.info(
                    "weather fetch: archive backfill for (%s, %s) %s..%s "
                    "(%d all-null forecast rows)",
                    lat, lon, run_start.isoformat(), run_end.isoformat(),
                    (run_end - run_start).days + 1,
                )
                backfill_df = _frame_from_payload(
                    _request_daily(
                        lat, lon, run_start.isoformat(), run_end.isoformat()
                    )
                )
                # Overlay: replace the null rows for these dates.
                df = (
                    pd.concat([df, backfill_df], ignore_index=True)
                    .drop_duplicates(subset=["date"], keep="last")
                    .sort_values("date")
                    .reset_index(drop=True)
                )

    return df


def _contiguous_runs(dates: list[dt.date]) -> list[tuple[dt.date, dt.date]]:
    """Compress a sorted list of dates into contiguous (start, end) runs."""
    if not dates:
        return []
    runs: list[tuple[dt.date, dt.date]] = []
    run_start = dates[0]
    prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days == 1:
            prev = d
            continue
        runs.append((run_start, prev))
        run_start = d
        prev = d
    runs.append((run_start, prev))
    return runs


def fetch_one(
    station_id: str,
    lat: float,
    lon: float,
    start: str,
    end: str,
    out_dir: Path,
    *,
    force: bool = False,
    forecast_only: bool = False,
) -> Path | None:
    """Fetch and cache weather for a single station. Returns the cached path.

    Uses the hybrid archive+forecast strategy (spec §13.7) — see
    :func:`_fetch_hybrid` for the per-range routing logic.

    Args:
        forecast_only: passed through to :func:`_fetch_hybrid`. When True,
            the 2016 ERA5 fallback is skipped — output has no rows before
            ``WEATHER_FORECAST_COVERAGE_START``.
    """
    out_path = out_dir / f"{station_id}.parquet"
    if not force and _cache_covers(out_path, start, end, forecast_only=forecast_only):
        logger.debug("cache hit %s — covers %s..%s", out_path, start, end)
        return out_path

    df = _fetch_hybrid(lat, lon, start, end, forecast_only=forecast_only)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Atomic write: write to .tmp then rename. Protects against mid-process
    # kill leaving a half-written parquet that _cache_covers might mistake
    # for a valid cache entry. The os-level rename is atomic on the same
    # filesystem (which is the case here — both paths in out_dir).
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
    tmp_path.replace(out_path)  # atomic on same FS; works on Windows too
    logger.info("wrote %d days for %s to %s", len(df), station_id, out_path)
    return out_path


def fetch(
    stations_path: Path,
    start: str,
    end: str,
    out_dir: Path,
    *,
    force: bool = False,
    inter_call_seconds: float = DEFAULT_INTER_CALL_SECONDS,
    forecast_only: bool = False,
) -> None:
    """Fetch daily weather for every (lat, lon) in `stations_path`.

    Args:
        stations_path: parquet with at least `station_id, lat, lon`.
        start: ISO date, inclusive.
        end: ISO date, inclusive. Clamped to yesterday if it's today or
            later (Open-Meteo publication lag).
        out_dir: directory for per-station `<station_id>.parquet` files.
        force: re-fetch ignoring cache.
        inter_call_seconds: delay between station calls. Default 0.1s
            keeps us well under Open-Meteo's free-tier rate limit.
        forecast_only: skip the 2016 ERA5 fallback; only hit the forecast
            API. Output parquets start at WEATHER_FORECAST_COVERAGE_START.
    """
    end = _clamp_end_to_yesterday(end)

    stations = pd.read_parquet(stations_path, columns=["station_id", "lat", "lon"])
    n_total = len(stations)
    usable = stations[stations["lat"].notna() & stations["lon"].notna()].reset_index(drop=True)
    n_usable = len(usable)
    n_skipped = n_total - n_usable
    if n_skipped:
        logger.warning("skipping %d stations with missing lat/lon", n_skipped)

    logger.info("fetching weather for %d stations (%s..%s)", n_usable, start, end)

    fetched = 0
    cached = 0
    failed = 0
    for i, row in enumerate(usable.itertuples(index=False), start=1):
        station_id = str(row.station_id)
        out_path = out_dir / f"{station_id}.parquet"
        if not force and _cache_covers(out_path, start, end, forecast_only=forecast_only):
            cached += 1
            continue

        # `row` comes from `usable.itertuples`; lat/lon are float64 columns.
        # mypy can't see that, so cast through string.
        lat = float(row.lat)  # type: ignore[arg-type]
        lon = float(row.lon)  # type: ignore[arg-type]

        try:
            fetch_one(
                station_id=station_id,
                lat=lat,
                lon=lon,
                start=start,
                end=end,
                out_dir=out_dir,
                force=force,
                forecast_only=forecast_only,
            )
            fetched += 1
        except Exception:
            failed += 1
            logger.exception(
                "failed to fetch weather for %s (%s, %s)", station_id, lat, lon
            )

        if i % 100 == 0:
            logger.info("progress: %d / %d (fetched=%d cached=%d failed=%d)",
                        i, n_usable, fetched, cached, failed)

        if inter_call_seconds > 0 and not force:
            time.sleep(inter_call_seconds)

    logger.info(
        "weather fetch complete: total=%d fetched=%d cached=%d skipped_no_latlon=%d failed=%d",
        n_total, fetched, cached, n_skipped, failed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", required=True, type=Path,
                        help="Parquet with station_id, lat, lon")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--inter-call-seconds", type=float, default=DEFAULT_INTER_CALL_SECONDS)
    parser.add_argument(
        "--forecast-only", action="store_true",
        help="Skip the 2016 ERA5 fallback — use the forecast API only. "
             "Output parquets start at WEATHER_FORECAST_COVERAGE_START.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    fetch(
        args.stations,
        args.start,
        args.end,
        args.out,
        force=args.force,
        inter_call_seconds=args.inter_call_seconds,
        forecast_only=args.forecast_only,
    )


if __name__ == "__main__":
    main()
