"""Fetch RBA Inflation Expectations from RBA G3.

Source: https://www.rba.gov.au/statistics/historical-data.html#inflation
URL:    https://www.rba.gov.au/statistics/tables/csv/g3-data.csv

Why this and not Roy Morgan Consumer Confidence (per the spec hint
in §5.2): the ANZ-Roy Morgan series doesn't publish a clean
machine-readable feed (no API, no CSV/XLS download, HTML tables only;
Roy Morgan also gates the underlying historical series behind a
commercial offering at store.roymorgan.com). RBA G3 covers the same
*signal* (consumer macro mood) with a clean quarterly CSV that goes
back to 1985. The substitution is documented in spec §5.2 + §7.4.

Granularity: quarterly. The feature builder forward-fills to daily —
see spec §7.4 (`ctx_inflation_expectations_lag_7`).

Coverage: 1985-12-31 → present (consumer series GCONEXP).

Output schema: ``date`` (date), ``inflation_expectations`` (float, percent).

We extract the Consumer series (`GCONEXP`) by default; the file also
exposes business / union / market / break-even series — additional
fetchers or a `series_id` arg could surface those if useful.

Spec: spec.md §5.2, §7.4.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from fuel_pred.fetch._ckan import download_bytes
from fuel_pred.fetch._rba import parse_rba_table

logger = logging.getLogger(__name__)

URL: str = "https://www.rba.gov.au/statistics/tables/csv/g3-data.csv"

# RBA mnemonic for the Consumer Inflation Expectations series in G3
# (1-year ahead, end-quarter observation, MI = Melbourne Institute survey).
SERIES_ID: str = "GCONEXP"

VALUE_COLUMN: str = "inflation_expectations"


def _is_cache_fresh(out: Path, max_age_days: float) -> bool:
    if not out.exists():
        return False
    age_days = (time.time() - out.stat().st_mtime) / 86400.0
    return age_days < max_age_days


def fetch(
    start: str,
    end: str,
    out: Path,
    *,
    force: bool = False,
    max_age_days: float = 14.0,
) -> None:
    """Fetch RBA inflation expectations and write Parquet
    ``date, inflation_expectations``.

    Args:
        start: ISO date, inclusive.
        end: ISO date, inclusive.
        out: output Parquet path.
        force: re-fetch ignoring cache.
        max_age_days: skip re-fetch when cache is fresher than this. Default
            14 days because G3 only updates quarterly.
    """
    if not force and _is_cache_fresh(out, max_age_days):
        logger.info("cache hit %s (< %.0f days old) — skipping fetch", out, max_age_days)
        return

    payload = download_bytes(URL)
    df = parse_rba_table(payload, URL, SERIES_ID, value_column_name=VALUE_COLUMN)

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"no inflation-expectations rows in range {start}..{end}")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d rows to %s", len(df), out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-age-days", type=float, default=14.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    fetch(args.start, args.end, args.out, force=args.force, max_age_days=args.max_age_days)


if __name__ == "__main__":
    main()
