"""Fetch daily weather aggregates from Open-Meteo for each station lat/lon.

Source: Historical Weather API (ERA5 reanalysis):
    https://archive-api.open-meteo.com/v1/archive

Granularity: daily, with day boundaries in `Australia/Sydney` local time so
that the resulting `date` column joins cleanly to the FuelCheck-derived
`date` column in `fuel_daily.parquet` (also a local-date).

Coverage: 2016-09 → present. Open-Meteo's archive serves ERA5 with a
~5-day publication lag; the API returns ERA5T (preliminary) for very
recent dates. We don't ask for dates in the future.

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

## Leakage caveat (spec §7.6)

ERA5 is a *reanalysis* — its values for date `t` are computed
retrospectively from observations gathered after `t`. Using the value
for date `t` to predict price[`t+1`] is therefore using future
information that wasn't available on day `t`. v1 accepts this as a
methodological compromise; the README must call it out, and v2 should
switch to Open-Meteo's Previous Runs API (lead-time = 1 day) for the
2024+ portion. The fetcher itself just pulls archive data — the
discipline of joining wx_* features at day `t` to predict `t+1` lives
in the feature builder.

Spec: spec.md §5.1, §7.6.
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
from tenacity import retry, stop_after_attempt, wait_exponential

from fuel_pred import config

logger = logging.getLogger(__name__)

ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"

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

# Polite delay between station calls. Open-Meteo's free tier allows
# ~600 calls/min; 0.1s = 600/min upper bound. We're well under.
DEFAULT_INTER_CALL_SECONDS: float = 0.1


@retry(
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=config.RETRY_BACKOFF_SECONDS, max=30),
    reraise=True,
)
def _request_daily(lat: float, lon: float, start: str, end: str) -> dict[str, Any]:
    """One Open-Meteo archive call. Retries on transient errors."""
    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARIABLES.keys()),
        "timezone": TIMEZONE,
    }
    response = requests.get(
        ARCHIVE_URL,
        params=params,
        headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"},
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    if "error" in body and body.get("error"):
        raise RuntimeError(f"Open-Meteo error for ({lat}, {lon}): {body.get('reason', body)}")
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


def _cache_covers(path: Path, start: str, end: str) -> bool:
    """Return True if the cached parquet covers the full requested range."""
    if not path.exists():
        return False
    try:
        cached = pd.read_parquet(path, columns=["date"])
    except Exception as exc:  # pragma: no cover — corrupt cache, force re-fetch
        logger.warning("could not read cache %s (%s) — re-fetching", path, exc)
        return False
    if cached.empty:
        return False
    cached_min = pd.to_datetime(cached["date"].min()).date()
    cached_max = pd.to_datetime(cached["date"].max()).date()
    requested_start = dt.date.fromisoformat(start)
    requested_end = dt.date.fromisoformat(end)
    return bool(cached_min <= requested_start and cached_max >= requested_end)


def _clamp_end_to_yesterday(end: str) -> str:
    """ERA5 has a ~5-day lag. Asking for today/tomorrow returns nulls or 400s.

    We clamp `end` to yesterday (in Sydney local time) so callers don't have
    to do their own date arithmetic. Logged as INFO when clamping happens.
    """
    requested = dt.date.fromisoformat(end)
    yesterday = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1)
    if requested > yesterday:
        logger.info("clamping end %s -> %s (ERA5 publication lag)", end, yesterday.isoformat())
        return yesterday.isoformat()
    return end


def fetch_one(
    station_id: str,
    lat: float,
    lon: float,
    start: str,
    end: str,
    out_dir: Path,
    *,
    force: bool = False,
) -> Path | None:
    """Fetch and cache weather for a single station. Returns the cached path."""
    out_path = out_dir / f"{station_id}.parquet"
    if not force and _cache_covers(out_path, start, end):
        logger.debug("cache hit %s — covers %s..%s", out_path, start, end)
        return out_path

    payload = _request_daily(lat, lon, start, end)
    df = _frame_from_payload(payload)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
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
) -> None:
    """Fetch daily weather for every (lat, lon) in `stations_path`.

    Args:
        stations_path: parquet with at least `station_id, lat, lon`.
        start: ISO date, inclusive.
        end: ISO date, inclusive. Clamped to yesterday if it's today or
            later (ERA5 publication lag).
        out_dir: directory for per-station `<station_id>.parquet` files.
        force: re-fetch ignoring cache.
        inter_call_seconds: delay between station calls. Default 0.1s
            keeps us well under Open-Meteo's free-tier rate limit.
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
        if not force and _cache_covers(out_path, start, end):
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    fetch(
        args.stations,
        args.start,
        args.end,
        args.out,
        force=args.force,
        inter_call_seconds=args.inter_call_seconds,
    )


if __name__ == "__main__":
    main()
