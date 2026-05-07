"""Fetch NSW FuelCheck monthly price-history CSVs.

Source: https://data.nsw.gov.au/data/dataset/fuel-check
Granularity: per price-update event (multiple per station per day)
Coverage: 2016-09 → present, monthly CSVs

The dataset page contains one resource per month. The stable resource ID
schema and the download URL pattern can be discovered via the CKAN package
endpoint:

    https://data.nsw.gov.au/data/api/3/action/package_show?id=fuel-check

Per CLAUDE.md §"Network etiquette":
- Set User-Agent
- Use tenacity for retries
- Cache to data/raw/fuelcheck/ — one parquet per month, content-addressable
- Skip re-fetch if file exists and is newer than --max-age-days

Schema of the raw CSVs (per spec.md §6.1 — verify against actual files):
    ServiceStationName, Address, Suburb, Postcode, Brand,
    FuelCode, PriceUpdatedDate, Price

NOTE: Schema has been observed to drift over time (column renames, extra
columns). The cleaner — not this fetcher — is responsible for normalisation.
This fetcher only writes raw bytes to local cache.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch(start: str, end: str, out_dir: Path, *, force: bool = False) -> None:
    """Fetch FuelCheck monthly CSVs into `out_dir`.

    Args:
        start: ISO date — only fetch months whose data covers this date or later.
        end: ISO date — only fetch months whose data covers this date or earlier.
        out_dir: directory to write `<YYYY-MM>.csv` files to. Created if missing.
        force: if True, re-download even when cached file exists.
    """
    raise NotImplementedError("TODO: implement per spec.md §5.1 + this module's docstring.")


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
