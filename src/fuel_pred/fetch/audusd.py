"""Fetch AUD/USD daily exchange rate from RBA F11.1.

Source: https://www.rba.gov.au/statistics/historical-data.html#exchange-rates

The spec hint described F11.1 as two CSVs ("1983-2009" and "2010-current").
Reality (as of 2026) differs:
  - 2023 → current is the only CSV.
  - 1983 → 2022 is published as 10 separate legacy ``.xls`` files
    (``YYYY-YYYY.xls``).

We fetch the 2023-current CSV and the two XLS files that overlap the
project span (2014-2017 and 2018-2022). Older XLS files are intentionally
omitted because the project only needs data from 2016-09 onwards.

TODO(spec): the spec says F11.1 is "two CSVs". This is no longer true.
Either amend the spec to describe the current source layout, or pin a
different upstream that provides the full history as a single feed.

Granularity: business days only.
Coverage: 2014-01-02 → present (sufficient for the project's 2016-09 span).

Output schema: ``date`` (date), ``audusd`` (float — USD per 1 AUD).

Spec: spec.md §5.1.
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import time
from pathlib import Path

import pandas as pd

from fuel_pred.fetch._ckan import download_bytes

logger = logging.getLogger(__name__)

# RBA's stable URLs (verified May 2026 from
# https://www.rba.gov.au/statistics/historical-data.html#exchange-rates).
URL_HIST_2014_2017: str = "https://www.rba.gov.au/statistics/tables/xls-hist/2014-2017.xls"
URL_HIST_2018_2022: str = "https://www.rba.gov.au/statistics/tables/xls-hist/2018-2022.xls"
URL_CURRENT: str = "https://www.rba.gov.au/statistics/tables/csv/f11.1-data.csv"

SOURCES: tuple[str, ...] = (URL_HIST_2014_2017, URL_HIST_2018_2022, URL_CURRENT)

# RBA's column mnemonic for AUD/USD in the F11.1 dataset.
SERIES_ID_AUDUSD: str = "FXRUSD"


def _is_cache_fresh(out: Path, max_age_days: float) -> bool:
    if not out.exists():
        return False
    age_days = (time.time() - out.stat().st_mtime) / 86400.0
    return age_days < max_age_days


def _read_rba_table(payload: bytes, url: str) -> pd.DataFrame:
    """Read an RBA F-series payload into a raw, header-less DataFrame.

    Picks ``read_csv`` or ``read_excel`` based on the URL extension.
    """
    if url.lower().endswith(".csv"):
        text = payload.decode("utf-8-sig")
        # The RBA CSV has a single-cell title row followed by metadata rows
        # and data rows with many columns. pandas' parsers (both engines)
        # struggle with the variable column count, so we read with stdlib
        # csv and right-pad each row to a uniform width.
        rows = list(csv.reader(io.StringIO(text)))
        width = max((len(r) for r in rows), default=0)
        padded = [r + [""] * (width - len(r)) for r in rows]
        return pd.DataFrame(padded, dtype=str)
    return pd.read_excel(
        io.BytesIO(payload), header=None, dtype=str, keep_default_na=False, engine="xlrd"
    )


def _parse_rba_table(payload: bytes, url: str, series_id: str) -> pd.DataFrame:
    """Extract a (date, value) frame for one series from an RBA F-series payload."""
    raw = _read_rba_table(payload, url)

    # Find the "Series ID" header row. Both CSV and XLS layouts have a row
    # whose first cell is "Series ID" with the per-column codes alongside.
    first_col = raw.iloc[:, 0].astype(str).str.strip().str.casefold()
    matches = first_col.index[first_col == "series id"].tolist()
    if not matches:
        raise RuntimeError(f"could not find 'Series ID' row in {url}")
    series_row: int = int(matches[0])

    # Locate the column index that holds our series.
    series_row_values = raw.iloc[series_row].astype(str).str.strip()
    value_col: int = -1
    for col_idx in range(raw.shape[1]):
        if str(series_row_values.iloc[col_idx]) == series_id:
            value_col = col_idx
            break
    if value_col < 0:
        raise RuntimeError(f"series {series_id!r} not present in {url}")

    body = raw.iloc[series_row + 1 :].copy()
    date_series = pd.to_datetime(body.iloc[:, 0], errors="coerce", dayfirst=True)
    value_series = pd.to_numeric(body.iloc[:, value_col], errors="coerce")

    out = pd.DataFrame({"date": date_series, "audusd": value_series})
    out = out[out["date"].notna() & out["audusd"].notna()].copy()
    out["date"] = out["date"].dt.date

    return out.reset_index(drop=True)


# Backwards-compatible alias used by the test suite.
_parse_rba_csv = _parse_rba_table


def fetch(
    start: str,
    end: str,
    out: Path,
    *,
    force: bool = False,
    max_age_days: float = 1.0,
) -> None:
    """Fetch AUD/USD daily and write Parquet with columns ``date, audusd``.

    Args:
        start: ISO date, inclusive.
        end: ISO date, inclusive.
        out: output Parquet path. Parent directory is created if missing.
        force: re-fetch ignoring cache.
        max_age_days: skip re-fetch when cache file is younger than this.
    """
    if not force and _is_cache_fresh(out, max_age_days):
        logger.info("cache hit %s (< %.2f days old) — skipping fetch", out, max_age_days)
        return

    frames: list[pd.DataFrame] = []
    for url in SOURCES:
        payload = download_bytes(url)
        frames.append(_parse_rba_table(payload, url, SERIES_ID_AUDUSD))

    df = pd.concat(frames, ignore_index=True)

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"no AUD/USD rows in range {start}..{end}")

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
