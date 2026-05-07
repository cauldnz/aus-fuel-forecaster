"""Fetch daily weather aggregates from Open-Meteo for each station lat/lon.

Sources:
- Historical Weather API (ERA5 reanalysis):
    https://archive-api.open-meteo.com/v1/archive
- (v2) Previous Runs API for forecast-leading weather:
    https://previous-runs-api.open-meteo.com/...

v1 uses Historical Weather across the full span — see spec.md §7.6 for the
methodological caveat.

Granularity: daily aggregates.
Coverage: 2016-09 → present.

Per-station caching: data/raw/weather/<station_id>.parquet.

Variables returned (spec.md §7.6):
    temperature_2m_max, temperature_2m_min, precipitation_sum,
    wind_speed_10m_max, weather_code

Spec: spec.md §5.1, §7.6.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch(stations_path: Path, start: str, end: str, out_dir: Path, *, force: bool = False) -> None:
    """Fetch daily weather for each (lat, lon) in stations_path.

    Args:
        stations_path: parquet with at least `station_id, lat, lon`.
        start: ISO date.
        end: ISO date.
        out_dir: per-station parquets here.
        force: re-fetch ignoring cache.
    """
    raise NotImplementedError("TODO: implement per spec.md §5.1 + §7.6.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", required=True, type=Path,
                        help="Parquet with station_id, lat, lon")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    fetch(args.stations, args.start, args.end, args.out, force=args.force)


if __name__ == "__main__":
    main()
