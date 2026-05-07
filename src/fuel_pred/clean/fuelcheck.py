"""Clean and aggregate raw FuelCheck monthly CSVs to daily granularity.

Reads: data/raw/fuelcheck/<YYYY-MM>.csv  (or .parquet, post-Phase 1)
Writes:
    - data/interim/fuel_daily.parquet  (daily aggregates per station × fuel)
    - data/interim/stations.parquet    (one row per unique station)

Tasks:
1. Concatenate monthly files. Schema may drift over time — handle gracefully.
2. Standardise brand strings via data/static/brand_aliases.csv.
3. Generate stable `station_id` as a hash of (name, address, suburb, postcode).
4. Aggregate intraday price events to daily mean/min/max/n per (station, fuel).
5. Build the unique stations roster (latest values per station_id).

Spec: spec.md §6.1, §6.2.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def clean(in_dir: Path, out_path: Path, stations_out: Path) -> None:
    """Concatenate, dedupe, standardise, aggregate.

    Args:
        in_dir: directory of raw FuelCheck monthly CSVs/parquets.
        out_path: where to write `fuel_daily.parquet`.
        stations_out: where to write `stations.parquet`.
    """
    raise NotImplementedError("TODO: implement per spec.md §6.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stations-out", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    clean(args.in_dir, args.out, args.stations_out)


if __name__ == "__main__":
    main()
