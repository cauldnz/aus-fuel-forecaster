"""Real-network smoke test for spatial.resolve_addrs.

Hits live G-NAF (remote mode) + Nominatim with a small subset of stations
so we catch real-world surprises without burning Nominatim's rate limit
on thousands of rows. Mirrors `verify_real_fetches.py` from Phase 1.

Usage:

    uv run python tools/verify_real_geocode.py                  # 5 stations
    uv run python tools/verify_real_geocode.py --limit 20       # more
    uv run python tools/verify_real_geocode.py --stations PATH  # custom file

Reads `data/interim/stations.parquet` by default (output of
`clean.fuelcheck`). Writes the geocoded subset to a tempdir — does
*not* mutate the real `data/interim/`.

Exit code: 0 if at least one station resolves successfully, 1 otherwise.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

from fuel_pred import config
from fuel_pred.spatial import resolve_addrs

logger = logging.getLogger("verify_real_geocode")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stations",
        type=Path,
        default=config.DATA_INTERIM / "stations.parquet",
        help="Path to a stations parquet (default: data/interim/stations.parquet)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of rows to geocode (default: 5)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.stations.exists():
        logger.error("stations file not found: %s — run clean.fuelcheck first", args.stations)
        return 1

    df = pd.read_parquet(args.stations)
    if len(df) == 0:
        logger.error("stations file is empty: %s", args.stations)
        return 1

    subset = df.head(args.limit).copy()
    # Strip any pre-existing geocoded columns so this is a fresh real-network test.
    for col in resolve_addrs.GEOCODED_COLUMNS:
        if col in subset.columns:
            subset[col] = pd.NA

    print(f"\n>>> geocoding first {len(subset)} stations from {args.stations}\n")

    with tempfile.TemporaryDirectory(prefix="verify_real_geocode_") as tmp:
        in_path = Path(tmp) / "stations_in.parquet"
        out_path = Path(tmp) / "stations_out.parquet"
        cache_dir = Path(tmp) / "cache"
        subset.to_parquet(in_path, engine="pyarrow", compression="zstd", index=False)

        resolve_addrs.resolve(in_path, out_path, cache_dir=cache_dir, force=True)

        result = pd.read_parquet(out_path)
        cols = ["name", "address", "suburb", "postcode", "lat", "lon", "geocoder", "mb_code"]
        cols = [c for c in cols if c in result.columns]
        print("\n=== results ===")
        print(result[cols].to_string(index=False))
        print()

        n_ok = int(result["lat"].notna().sum())
        breakdown = result["geocoder"].value_counts(dropna=False).to_dict()
        print(f"resolved: {n_ok} / {len(result)}")
        print(f"by provider: {breakdown}")

    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
