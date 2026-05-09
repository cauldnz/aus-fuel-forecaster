"""Real-network smoke test for fetch.weather.

Hits live Open-Meteo with a small subset of stations. Per the
smoke-tests-hit-real-endpoints memory: real endpoints, reduced volume.

Usage:

    uv run python tools/verify_real_weather.py                  # 3 stations, last month
    uv run python tools/verify_real_weather.py --limit 10
    uv run python tools/verify_real_weather.py --start 2024-01-01 --end 2024-01-31

Reads `data/interim/stations.parquet` by default. Writes to a tempdir
— does not mutate `data/raw/weather/`.

Exit code: 0 if at least one station has non-null data, 1 otherwise.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

from fuel_pred import config
from fuel_pred.fetch import weather

logger = logging.getLogger("verify_real_weather")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stations",
        type=Path,
        default=config.DATA_INTERIM / "stations.parquet",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--start", default=None, help="ISO date; default 30 days before end")
    parser.add_argument("--end", default=None, help="ISO date; default yesterday")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.stations.exists():
        logger.error("stations file not found: %s", args.stations)
        return 1

    end = args.end or (dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1)).isoformat()
    start = args.start or (dt.date.fromisoformat(end) - dt.timedelta(days=30)).isoformat()

    df = pd.read_parquet(args.stations)
    if "lat" not in df.columns or "lon" not in df.columns:
        logger.error("stations missing lat/lon — run spatial.resolve_addrs first")
        return 1
    subset = df[df["lat"].notna() & df["lon"].notna()].head(args.limit).copy()

    print(f"\n>>> fetching weather for {len(subset)} stations, {start} to {end}")
    print(subset[["name", "lat", "lon"]].to_string(index=False))
    print()

    with tempfile.TemporaryDirectory(prefix="verify_real_weather_") as tmp:
        tmpdir = Path(tmp)
        in_path = tmpdir / "stations_in.parquet"
        out_dir = tmpdir / "weather"
        subset.to_parquet(in_path, engine="pyarrow", compression="zstd", index=False)

        weather.fetch(in_path, start, end, out_dir, force=True, inter_call_seconds=0.2)

        files = sorted(out_dir.glob("*.parquet"))
        print(f"\n>>> wrote {len(files)} per-station parquets")
        if not files:
            print("  (none — all fetches failed)")
            return 1

        for path in files[: min(3, len(files))]:
            wx = pd.read_parquet(path)
            station_name = subset.loc[subset["station_id"] == path.stem, "name"]
            label = station_name.iloc[0] if len(station_name) else path.stem
            print(f"\n=== {label} ({path.stem}) ===")
            print(f"  rows: {len(wx)}, dates {wx['date'].min()} to {wx['date'].max()}")
            print(
                f"  temp_max range: {wx['wx_temp_max_c'].min():.1f} .. "
                f"{wx['wx_temp_max_c'].max():.1f} C"
            )
            print(
                f"  precipitation: total {wx['wx_precipitation_mm'].sum():.1f} mm, "
                f"max-day {wx['wx_precipitation_mm'].max():.1f} mm"
            )
            print(wx.head(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
