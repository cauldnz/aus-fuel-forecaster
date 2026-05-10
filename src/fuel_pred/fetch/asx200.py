"""Fetch ASX 200 daily close via yfinance (ticker ^AXJO).

Source: Yahoo Finance via the `yfinance` package.
Granularity: daily, business days.
Coverage: 1992 → present (Yahoo's ASX 200 history).

Mirrors `fetch.brent` — same yfinance pattern, different ticker, output
schema trimmed to what spec §7.4 needs (`date, close`).

Spec: spec.md §5.2.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TICKER: str = "^AXJO"
SCHEMA: tuple[str, ...] = ("date", "open", "high", "low", "close", "volume")


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
    max_age_days: float = 1.0,
) -> None:
    """Fetch daily ASX 200 OHLC and write Parquet with columns
    ``date, open, high, low, close, volume``.

    Args:
        start: ISO date, inclusive.
        end: ISO date, exclusive (yfinance convention).
        out: output Parquet path. Parent directory is created if missing.
        force: if True, re-fetch even when cache is fresh.
        max_age_days: skip re-fetch when cache file is younger than this.
    """
    if not force and _is_cache_fresh(out, max_age_days):
        logger.info("cache hit %s (< %.2f days old) — skipping fetch", out, max_age_days)
        return

    logger.info("fetching %s from yfinance: start=%s end=%s", TICKER, start, end)
    ticker = yf.Ticker(TICKER)
    df = ticker.history(start=start, end=end, auto_adjust=False)

    if df.empty:
        raise RuntimeError(f"yfinance returned empty frame for {TICKER} {start}..{end}")

    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    if "date" not in df.columns:
        raise RuntimeError(f"unexpected yfinance schema: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date

    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise RuntimeError(f"yfinance missing expected columns {missing}: got {list(df.columns)}")

    df = df.loc[:, list(SCHEMA)]

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d rows to %s", len(df), out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-age-days", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    fetch(args.start, args.end, args.out, force=args.force, max_age_days=args.max_age_days)


if __name__ == "__main__":
    main()
