"""Clean TfNSW traffic counts: hourly → daily, station reference normalised.

Reads:
    - data/raw/traffic/stations.parquet
    - data/raw/traffic/hourly.parquet
Writes:
    - data/interim/traffic_daily.parquet  (date, counter_id, vehicle_count)
    - data/interim/traffic_stations.parquet  (counter_id, lat, lon, road, suburb, postcode, quality)

Filtering: drop stations flagged as low-quality in the source reference table.

Spec: spec.md §5.1, §7.4.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def clean(in_dir: Path, out: Path, stations_out: Path) -> None:
    raise NotImplementedError("TODO: implement per spec.md §7.4.")


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
