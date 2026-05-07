"""Fetch NSW FuelCheck monthly price-history CSVs.

Source: https://data.nsw.gov.au/data/dataset/fuel-check
Granularity: per price-update event (multiple per station per day)
Coverage: 2016-09 → present, monthly CSVs

The dataset page contains one resource per month. Discover the resource list
via the CKAN package endpoint:

    https://data.nsw.gov.au/data/api/3/action/package_show?id=fuel-check

Per CLAUDE.md "Network etiquette":
- Set User-Agent
- Use tenacity for retries
- Cache to ``<out_dir>/<YYYY-MM>.parquet``
- Skip re-fetch if the parquet already exists, unless ``--force``

Schema: NOT enforced here. Schema has drifted over the years (column
renames, extra columns). Use ``pandas.read_csv(low_memory=False)`` and
write whatever we got — the cleaner is responsible for normalisation.
"""
from __future__ import annotations

import argparse
import io
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from fuel_pred.fetch import _ckan

logger = logging.getLogger(__name__)

API_ROOT: str = "https://data.nsw.gov.au/data/api/3/action"
PACKAGE_ID: str = "fuel-check"

# Match resources whose name suggests "price history" (case-insensitive).
PRICE_HISTORY_PATTERNS: tuple[str, ...] = ("price history", "price_history", "pricehistory")
# Service-station / brand reference resources we want to *exclude*.
EXCLUDE_PATTERNS: tuple[str, ...] = ("service station and brand", "service_station_and_brand")
# Tabular formats we know how to read. The NSW FuelCheck dataset uses a
# mix: ~94 .xlsx, 8 .csv (verified May 2026 via package_show).
SUPPORTED_FORMATS: tuple[str, ...] = (
    "csv",
    "text/csv",
    "xlsx",
    "excel (.xlsx)",
    "excel (xlsx)",
    "",  # empty format → trust the URL extension
)

_MONTH_NAMES: dict[str, int] = {
    name.lower(): i
    for i, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
_MONTH_ABBR: dict[str, int] = {
    name.lower(): i
    for i, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        start=1,
    )
}


def extract_year_month(text: str) -> tuple[int, int] | None:
    """Best-effort extraction of (year, month) from a resource name or URL.

    Recognises patterns like ``2024-08``, ``2024_08``, ``August 2024``,
    ``Aug-2024``, ``aug2024``, ``2024 08`` — covering the variants the
    NSW FuelCheck dataset has used over time.
    """
    cleaned = text.lower()

    # YYYY-MM / YYYY_MM / YYYY MM
    m = re.search(r"(?<!\d)(20\d{2})[-_ ](0?[1-9]|1[0-2])(?!\d)", cleaned)
    if m:
        return int(m.group(1)), int(m.group(2))

    # MM-YYYY (e.g. "08-2024") — accept but don't catch random pairs.
    m = re.search(r"(?<!\d)(0?[1-9]|1[0-2])[-_ ](20\d{2})(?!\d)", cleaned)
    if m:
        return int(m.group(2)), int(m.group(1))

    # "Month YYYY" or "Month-YYYY" (full or abbreviated).
    name_alternation = "|".join(list(_MONTH_NAMES.keys()) + list(_MONTH_ABBR.keys()))
    m = re.search(rf"({name_alternation})[-_\s]?(20\d{{2}})", cleaned)
    if m:
        word = m.group(1)
        year = int(m.group(2))
        month = _MONTH_NAMES.get(word) or _MONTH_ABBR.get(word)
        if month is not None:
            return year, month

    # "YYYY-Month"
    m = re.search(rf"(20\d{{2}})[-_\s]({name_alternation})", cleaned)
    if m:
        year = int(m.group(1))
        word = m.group(2)
        month = _MONTH_NAMES.get(word) or _MONTH_ABBR.get(word)
        if month is not None:
            return year, month

    return None


def _is_price_history_resource(resource: dict[str, Any]) -> bool:
    name = str(resource.get("name", "")).lower()
    fmt = str(resource.get("format", "")).lower()
    url = str(resource.get("url", "")).lower()
    # Reject when the format string isn't one we know AND the URL doesn't
    # carry a recognisable tabular extension.
    known_url = url.endswith(".csv") or url.endswith(".xlsx")
    if fmt not in SUPPORTED_FORMATS and not (fmt == "" and known_url):
        return False
    if any(ex in name for ex in EXCLUDE_PATTERNS):
        return False
    return any(p in name for p in PRICE_HISTORY_PATTERNS)


def _read_payload(payload: bytes, url: str) -> pd.DataFrame:
    """Parse a downloaded FuelCheck monthly file (CSV or XLSX) verbatim."""
    lower = url.lower()
    if lower.endswith(".xlsx"):
        df: pd.DataFrame = pd.read_excel(io.BytesIO(payload), engine="openpyxl")
        return df.astype(str)
    if lower.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(payload), engine="xlrd")
        return df.astype(str)
    return pd.read_csv(io.BytesIO(payload), low_memory=False, dtype=str)


def _resource_to_month(resource: dict[str, Any]) -> tuple[int, int] | None:
    name = str(resource.get("name", ""))
    url = str(resource.get("url", ""))
    return extract_year_month(name) or extract_year_month(url)


def _in_range(ym: tuple[int, int], start: str, end: str) -> bool:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    month_start = pd.Timestamp(year=ym[0], month=ym[1], day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    return not (month_end < start_dt or month_start > end_dt)


def fetch(start: str, end: str, out_dir: Path, *, force: bool = False) -> None:
    """Fetch FuelCheck monthly CSVs into ``out_dir`` as one parquet per month.

    Args:
        start: ISO date — only fetch months whose data covers this date or later.
        end: ISO date — only fetch months whose data covers this date or earlier.
        out_dir: directory to write ``<YYYY-MM>.parquet`` files. Created if missing.
        force: re-download even when cached file exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("looking up FuelCheck package %s at %s", PACKAGE_ID, API_ROOT)
    package = _ckan.package_show(API_ROOT, PACKAGE_ID)
    resources = package.get("resources", [])

    candidates: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for resource in resources:
        if not _is_price_history_resource(resource):
            continue
        ym = _resource_to_month(resource)
        if ym is None:
            logger.warning(
                "skipping price-history resource %r — could not parse month",
                resource.get("name"),
            )
            continue
        if not _in_range(ym, start, end):
            continue
        candidates.append((ym, resource))

    candidates.sort(key=lambda t: t[0])
    logger.info("matched %d monthly price-history resources in range", len(candidates))

    if not candidates:
        logger.warning("no FuelCheck price-history resources matched %s..%s", start, end)
        return

    fetched = 0
    skipped = 0
    for (year, month), resource in candidates:
        out_path = out_dir / f"{year:04d}-{month:02d}.parquet"
        if out_path.exists() and not force:
            logger.info("cache hit %s — skipping", out_path)
            skipped += 1
            continue

        url = str(resource["url"])
        payload = _ckan.download_bytes(url)
        df = _read_payload(payload, url)
        df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        logger.info("wrote %d rows to %s", len(df), out_path)
        fetched += 1

    logger.info("fuelcheck fetch complete: fetched=%d skipped=%d", fetched, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="ISO date, e.g. 2016-09-01")
    parser.add_argument("--end", required=True, help="ISO date, e.g. 2026-04-30")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Re-download cached files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    fetch(args.start, args.end, args.out, force=args.force)


if __name__ == "__main__":
    main()
