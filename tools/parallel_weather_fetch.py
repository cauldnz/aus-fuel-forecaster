"""Parallel orchestrator for the hybrid weather fetcher.

The serial `fetch.weather` module takes ~8.5s per station with the v2 hybrid
logic (1 forecast call + 1 archive call + safety-net backfill). For ~4,500
NSW stations that's ~10.6 hours single-threaded — unacceptable.

This script runs `fetch_one()` across a pool of workers. With 4 workers we
stay well under Open-Meteo's free-tier rate limit (600 req/min) since each
worker makes ~1 call/sec internally, totalling ~4-5 calls/sec across the
pool. Expected wall-clock for the full roster: ~2-3 hours.

Per-station progress is logged inline. Stations already cached are skipped
without an API call (matches the serial behaviour). Failures are logged and
counted but don't abort the run.

CLI:
    uv run python tools/parallel_weather_fetch.py \
        --stations data/interim/stations.parquet \
        --start 2016-09-01 --end 2026-04-30 \
        --out data/raw/weather --workers 4

Note: this is a `tools/` script per project convention (real-network
integration utilities). It's not a pipeline module — the canonical fetcher
is still `python -m fuel_pred.fetch.weather`, which this just parallelises.
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from fuel_pred.fetch.weather import _cache_covers, _clamp_end_to_yesterday, fetch_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parallel_weather_fetch")


def _fetch_worker(
    args: tuple[str, float, float, str, str, str, bool],
) -> tuple[str, bool, str]:
    """Fetch one station; return (station_id, success, message)."""
    station_id, lat, lon, start, end, out_dir, forecast_only = args
    try:
        fetch_one(
            station_id=station_id,
            lat=lat,
            lon=lon,
            start=start,
            end=end,
            out_dir=Path(out_dir),
            force=False,
            forecast_only=forecast_only,
        )
        return (station_id, True, "ok")
    except Exception as exc:
        return (station_id, False, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker process count (default 4; stays under API rate limit)")
    parser.add_argument("--progress-every", type=int, default=50,
                        help="Log progress every N completions")
    parser.add_argument(
        "--forecast-only", action="store_true",
        help="Skip 2016 ERA5 fallback — forecast API only. Output parquets "
             "start at WEATHER_FORECAST_COVERAGE_START (no pre-2017 rows).",
    )
    args = parser.parse_args()

    end = _clamp_end_to_yesterday(args.end)
    args.out.mkdir(parents=True, exist_ok=True)

    stations = pd.read_parquet(args.stations, columns=["station_id", "lat", "lon"])
    n_total = len(stations)
    usable = stations[stations["lat"].notna() & stations["lon"].notna()].reset_index(drop=True)
    n_skipped_no_latlon = n_total - len(usable)
    if n_skipped_no_latlon:
        logger.warning("skipping %d stations with missing lat/lon", n_skipped_no_latlon)

    # Skip stations that already have a cache covering the requested span.
    pending = []
    cached = 0
    for row in usable.itertuples(index=False):
        sid = str(row.station_id)
        if _cache_covers(args.out / f"{sid}.parquet", args.start, end):
            cached += 1
            continue
        pending.append(
            (sid, float(row.lat), float(row.lon), args.start, end, str(args.out),
             args.forecast_only)
        )

    logger.info(
        "weather fetch: total=%d cached=%d pending=%d workers=%d range=%s..%s forecast_only=%s",
        n_total, cached, len(pending), args.workers, args.start, end, args.forecast_only,
    )

    if not pending:
        logger.info("nothing to fetch — all stations already cached")
        return

    fetched = 0
    failed = 0
    started = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_fetch_worker, item) for item in pending]
        for i, fut in enumerate(as_completed(futures), start=1):
            sid, ok, msg = fut.result()
            if ok:
                fetched += 1
            else:
                failed += 1
                logger.warning("FAIL %s: %s", sid, msg)
            if i % args.progress_every == 0 or i == len(pending):
                elapsed = time.monotonic() - started
                rate = i / elapsed if elapsed > 0 else 0
                eta_min = (len(pending) - i) / rate / 60 if rate > 0 else 0
                logger.info(
                    "progress %d/%d  fetched=%d failed=%d  rate=%.2f stn/s  ETA=%.1f min",
                    i, len(pending), fetched, failed, rate, eta_min,
                )

    total_elapsed = time.monotonic() - started
    logger.info(
        "weather fetch complete: fetched=%d failed=%d cached=%d skipped_no_latlon=%d  total=%.1f min",
        fetched, failed, cached, n_skipped_no_latlon, total_elapsed / 60,
    )


if __name__ == "__main__":
    main()
