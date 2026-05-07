"""Resolve station addresses to lat/lon via G-NAF, falling back to Nominatim.

Reads `stations.parquet` (with at least `address, suburb, postcode`) and
writes the same parquet with `lat, lon, geocoder` columns populated.

G-NAF is the preferred resolver — much higher hit rate on real Australian
addresses than Nominatim, no rate limits when used locally. The
`abs-census-augmentor` package exposes a remote G-NAF lookup
(see https://github.com/cauldnz/abs-census-augmentor PR queue) which we
call here. Nominatim is the fallback for misses.

Spec: spec.md §6.1, §12 Phase 2.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve(in_path: Path, out_path: Path) -> None:
    """Add lat/lon/geocoder columns to the stations parquet (in-place safe)."""
    raise NotImplementedError("TODO: implement per spec.md Phase 2.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    resolve(args.in_path, args.out)


if __name__ == "__main__":
    main()
