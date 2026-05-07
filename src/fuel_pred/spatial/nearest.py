"""Compute nearest-neighbour assignments: each station → nearest traffic counter.

Output schema:
    station_id, counter_id, distance_km

Stations whose nearest counter is more than 50 km away should still appear in
the output, but downstream feature engineering treats `ctx_traffic_*` as null
when distance > 50 km (see spec.md §7.4).

Also computes `stn_distance_to_sydney_terminal_km` per station — Sydney
fuel terminal is at Botany (-33.9619, 151.2095). Hard-coded here because
Australian fuel terminal locations don't change.

Spec: spec.md §7.4, §7.5.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Sydney fuel terminal — Botany. Distance feature uses this anchor.
SYDNEY_TERMINAL_LAT: float = -33.9619
SYDNEY_TERMINAL_LON: float = 151.2095


def compute_nearest(stations_path: Path, counters_path: Path, out_path: Path) -> None:
    raise NotImplementedError("TODO: implement per spec.md §7.4 + §7.5.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", required=True, type=Path)
    parser.add_argument("--counters", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    compute_nearest(args.stations, args.counters, args.out)


if __name__ == "__main__":
    main()
