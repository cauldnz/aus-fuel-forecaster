"""Parallel orchestrator for the GFS/GEFS multi-horizon fetcher.

The serial `fetch.gfs` module takes ~2-4s per (date, horizon) end-to-end —
~5s of byte-range I/O across 4 leads, plus cfgrib parse + parquet write.
For the full backfill (3,529 dates × 7 horizons = 24,703 files) that's
~14-28 hours single-threaded — too slow.

This script runs `fetch_and_write_one_day()` across a pool of workers,
with the unit of work being a single (date, horizon). NOAA S3 is publicly
accessible with no rate limit we've hit in practice; 4 workers is a safe
default that won't saturate the user's bandwidth or DNS.

Per-(date, horizon) progress is logged inline. Files already on disk are
skipped without a network call (matches the serial behaviour). Failures
are logged + counted but don't abort the run — re-running picks up where
we left off (cache-aware).

CLI:
    uv run python tools/parallel_gfs_fetch.py \
        --start 2017-01-01 --end 2026-04-30 \
        --out data/raw/weather_gfs --horizons 1,2,3,4,5,6,7 --workers 4

Note: this is a `tools/` script per project convention (real-network
integration utilities). The canonical fetcher is still
`python -m fuel_pred.fetch.gfs`, which this just parallelises.

Unlike the Open-Meteo parallel fetcher, there's no rate-limit circuit
breaker — NOAA S3 doesn't throttle anonymous reads. The only realistic
failure mode is transient network hiccup; tenacity handles retries
inside the fetcher.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fuel_pred.fetch.gfs import (
    DEFAULT_CYCLE,
    DEFAULT_HORIZONS,
    _grid_parquet_path,
    fetch_and_write_one_day,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parallel_gfs_fetch")


def _fetch_worker(
    args: tuple[str, int, str, str, str, bool],
) -> tuple[str, int, bool, str]:
    """Fetch one (date, horizon); return (date_iso, horizon, success, message)."""
    date_iso, horizon, cycle, out_dir, _log_label, force = args
    try:
        fetch_and_write_one_day(
            date=dt.date.fromisoformat(date_iso),
            out_dir=Path(out_dir),
            horizons=(horizon,),
            cycle=cycle,
            force=force,
        )
        return (date_iso, horizon, True, "ok")
    except Exception as exc:  # broad: log everything, no abort
        return (date_iso, horizon, False, str(exc))


def _pending_units(
    start: dt.date, end: dt.date,
    horizons: tuple[int, ...],
    out_dir: Path,
    cycle: str,
    *, force: bool,
) -> tuple[list[tuple[str, int, str, str, str, bool]], int]:
    """Build the list of (date, horizon) work items, skipping cached ones."""
    pending: list[tuple[str, int, str, str, str, bool]] = []
    cached = 0
    n_days = (end - start).days + 1
    for n in range(n_days):
        d = start + dt.timedelta(days=n)
        for h in horizons:
            out_path = _grid_parquet_path(out_dir, d, h)
            if not force and out_path.exists():
                cached += 1
                continue
            pending.append((
                d.isoformat(), h, cycle, str(out_dir), f"{d.isoformat()}_h{h}", force,
            ))
    return pending, cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True,
                        help="ISO date (YYYY-MM-DD) — inclusive.")
    parser.add_argument("--end", required=True,
                        help="ISO date (YYYY-MM-DD) — inclusive.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output dir for per-(date, horizon) grid parquets.")
    parser.add_argument(
        "--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated horizon days (1..7). Default: 1..7.",
    )
    parser.add_argument("--cycle", default=DEFAULT_CYCLE,
                        choices=["00", "06", "12", "18"])
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker process count. Default 4 — safe for S3.")
    parser.add_argument("--progress-every", type=int, default=50,
                        help="Log progress every N completions.")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch ignoring cache.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    if start > end:
        raise SystemExit(f"--start ({args.start}) must be <= --end ({args.end})")

    horizons = tuple(int(x) for x in args.horizons.split(","))

    pending, cached = _pending_units(
        start, end, horizons, args.out, args.cycle, force=args.force,
    )
    total = (end - start).days + 1
    logger.info(
        "gfs fetch: dates=%d × horizons=%s = %d units; cached=%d pending=%d workers=%d",
        total, list(horizons), total * len(horizons), cached, len(pending), args.workers,
    )

    if not pending:
        logger.info("nothing to fetch — every (date, horizon) is cached")
        return

    fetched = 0
    failed = 0
    started = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_fetch_worker, item) for item in pending]
        for i, fut in enumerate(as_completed(futures), start=1):
            date_iso, horizon, ok, msg = fut.result()
            if ok:
                fetched += 1
            else:
                failed += 1
                logger.warning("FAIL %s h=%d: %s", date_iso, horizon, msg)

            if i % args.progress_every == 0 or i == len(pending):
                elapsed = time.monotonic() - started
                rate = i / elapsed if elapsed > 0 else 0
                eta_min = (len(pending) - i) / rate / 60 if rate > 0 else 0
                logger.info(
                    "progress %d/%d  fetched=%d failed=%d  rate=%.2f units/s  "
                    "ETA=%.1f min",
                    i, len(pending), fetched, failed, rate, eta_min,
                )

    total_elapsed = time.monotonic() - started
    logger.info(
        "gfs fetch complete: fetched=%d failed=%d cached=%d total=%.1f min",
        fetched, failed, cached, total_elapsed / 60,
    )


if __name__ == "__main__":
    main()
