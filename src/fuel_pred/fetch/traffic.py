"""Fetch NSW Roads Traffic Volume Counts from TfNSW open-data portal.

Source: https://opendata.transport.nsw.gov.au/data/dataset/nsw-roads-traffic-volume-counts-api
        (CKAN datastore — multiple resources)
Granularity: hourly per station; clean.traffic aggregates to daily.
Coverage: 2006 → present (with documented data-quality issues).

Tables to fetch (spec.md §5.1 reference):
- Traffic Collection Station Reference  (lat/lon, road, suburb, postcode, quality)
- Permanent Hourly Traffic Counts        (the time series)

Cache one parquet per table under `out_dir`.

Spec: spec.md §5.1.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch(start: str, end: str, out_dir: Path, *, force: bool = False) -> None:
    """Fetch the traffic reference + hourly tables to `out_dir`.

    Args:
        start: ISO date — earliest day to include in hourly.
        end: ISO date — latest day to include in hourly.
        out_dir: written: `stations.parquet`, `hourly.parquet`.
        force: re-fetch ignoring cache.
    """
    raise NotImplementedError("TODO: implement per spec.md §5.1.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    fetch(args.start, args.end, args.out, force=args.force)


if __name__ == "__main__":
    main()
