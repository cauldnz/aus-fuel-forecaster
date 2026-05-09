"""Real-network smoke test for build.enrich_census.

Hits live ABS SEIFA + the augmentor's S3 boundary parquets with a small
subset of stations so we catch real-world surprises early. Mirrors the
verify_real_geocode.py / verify_real_fetches.py pattern.

Usage:

    uv run python tools/verify_real_enrich.py                  # 5 stations
    uv run python tools/verify_real_enrich.py --limit 20

Reads `data/interim/stations.parquet` (output of `clean.fuelcheck` +
`spatial.resolve_addrs`). Writes the enriched subset to a tempdir —
does *not* mutate the real `data/interim/`.

Exit code: 0 if at least one station enriches successfully, 1 otherwise.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

from fuel_pred import config
from fuel_pred.build import enrich_census
from fuel_pred.fetch import seifa as fetch_seifa

logger = logging.getLogger("verify_real_enrich")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stations",
        type=Path,
        default=config.DATA_INTERIM / "stations.parquet",
        help="Path to a stations parquet (default: data/interim/stations.parquet)",
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.stations.exists():
        logger.error("stations file not found: %s", args.stations)
        return 1

    df = pd.read_parquet(args.stations)
    if "lat" not in df.columns or "lon" not in df.columns:
        logger.error(
            "stations missing lat/lon — run spatial.resolve_addrs first "
            "(or pass a post-Phase-2 file)"
        )
        return 1

    subset = df[df["lat"].notna() & df["lon"].notna()].head(args.limit).copy()
    # Strip any pre-existing enriched columns so we exercise the real path.
    for col in enrich_census.ENRICHED_COLUMNS:
        if col in subset.columns:
            subset[col] = pd.NA

    print(f"\n>>> enriching {len(subset)} stations ...")
    print(subset[["name", "lat", "lon"]].to_string(index=False))
    print()

    with tempfile.TemporaryDirectory(prefix="verify_real_enrich_") as tmp:
        tmpdir = Path(tmp)
        in_path = tmpdir / "stations_in.parquet"
        out_path = tmpdir / "stations_out.parquet"
        seifa_path = tmpdir / "seifa.parquet"

        subset.to_parquet(in_path, engine="pyarrow", compression="zstd", index=False)

        print(">>> fetching SEIFA ...")
        fetch_seifa.fetch(seifa_path, force=True)

        print(">>> calling augmentor (may take a minute on first run for boundary download) ...")
        enrich_census.enrich(in_path, out_path, seifa_path=seifa_path)

        result = pd.read_parquet(out_path)
        cols = [
            "name",
            "lat",
            "lon",
            "sa2_code",
            "sa2_name",
            "sa2_total_population",
            "sa2_median_age",
            "sa2_median_household_income_weekly",
            "sa2_seifa_irsd_score",
        ]
        cols = [c for c in cols if c in result.columns]
        print("\n=== results ===")
        print(result[cols].to_string(index=False))
        print()

        n_sa2 = int(result["sa2_code"].notna().sum())
        n_seifa = int(result["sa2_seifa_irsd_score"].notna().sum())
        print(f"sa2_code populated: {n_sa2} / {len(result)}")
        print(f"sa2_seifa_irsd_score populated: {n_seifa} / {len(result)}")

    return 0 if n_sa2 > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
