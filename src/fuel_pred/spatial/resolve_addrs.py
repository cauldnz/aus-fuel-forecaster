"""Resolve station addresses to lat/lon via G-NAF, falling back to Nominatim.

Reads `stations.parquet` (with at least `name, address, suburb, postcode`)
and writes the same parquet with `lat, lon, geocoder, mb_code` columns
populated.

Geocoding cascade (per spec.md §12 Phase 2):

1. **G-NAF (remote mode)** — `census_augment.GnafGeocoder` streams G-NAF
   parquet directly from S3 via DuckDB httpfs. No 10 GB local download.
   Authoritative for Australian addresses; high hit rate on real
   FuelCheck data.
2. **Nominatim** — only used for G-NAF misses. Hard-rate-limited to
   1 req/sec per Nominatim's usage policy. Disk-backed cache avoids
   re-querying for the same address on subsequent runs.

Idempotency: rows whose `(lat, lon, geocoder)` are already non-null
are skipped unless `--force`. Re-runs after partial completion (e.g.
crashed during Nominatim batch) only resolve the unfinished rows.

Geocoding per `station_id` — *not* per unique `(address, suburb,
postcode)` triple — because the user has decided to model each station
separately even when address strings collide.

Spec: spec.md §6.1, §12 Phase 2.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# These are runtime-public but the augmentor's submodules don't list them
# in `__all__`, so mypy strict needs the attribute-defined ignores.
from census_augment.geocoding.gnaf import (  # type: ignore[attr-defined]
    GnafDataSource,
    GnafGeocoder,
)
from census_augment.geocoding.nominatim import (  # type: ignore[attr-defined]
    GeocodeCache,
    NominatimGeocoder,
)

from fuel_pred import config

logger = logging.getLogger(__name__)

GEOCODED_COLUMNS: tuple[str, ...] = ("lat", "lon", "geocoder", "mb_code")


def _format_address(row: pd.Series) -> str:
    """Build a single address string from FuelCheck columns."""
    parts = [
        str(row.get("address", "") or ""),
        str(row.get("suburb", "") or ""),
        f"NSW {row.get('postcode', '') or ''}".strip(),
        "Australia",
    ]
    return ", ".join(p.strip() for p in parts if p.strip())


def _ensure_geocoded_columns(stations: pd.DataFrame) -> pd.DataFrame:
    """Make sure the expected output columns exist (as nullable types)."""
    out = stations.copy()
    for col in GEOCODED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _row_is_geocoded(row: pd.Series) -> bool:
    return pd.notna(row.get("lat")) and pd.notna(row.get("lon"))


def _build_geocoders(
    cache_dir: Path,
    user_agent: str,
    *,
    nominatim_factory: object | None = None,
    gnaf_factory: object | None = None,
) -> tuple[object, object]:
    """Construct the G-NAF (remote) and Nominatim geocoders.

    `nominatim_factory` and `gnaf_factory` exist so tests can inject
    in-memory fakes without hitting the network.
    """
    if gnaf_factory is None:
        data_source = GnafDataSource(
            mode="remote",
            release="latest",
            data_dir=cache_dir / "gnaf",
        )
        gnaf = GnafGeocoder(data_source=data_source)
    else:
        gnaf = gnaf_factory()  # type: ignore[operator]

    if nominatim_factory is None:
        nom_cache = GeocodeCache(root=cache_dir / "nominatim")
        nominatim = NominatimGeocoder(
            user_agent=user_agent,
            cache=nom_cache,
            rate_limit_per_second=1.0,
        )
    else:
        nominatim = nominatim_factory()  # type: ignore[operator]

    return gnaf, nominatim


def _geocode_one(address: str, gnaf: object, nominatim: object) -> dict[str, Any]:
    """Try G-NAF first, fall back to Nominatim. Returns row-update dict."""
    result = gnaf.geocode(address)  # type: ignore[attr-defined]
    if result.is_success:
        return {
            "lat": result.lat,
            "lon": result.lon,
            "geocoder": "gnaf",
            "mb_code": result.mb_code,
        }

    logger.debug("G-NAF miss for %r — falling back to Nominatim", address)
    fallback = nominatim.geocode(address)  # type: ignore[attr-defined]
    if fallback.is_success:
        return {
            "lat": fallback.lat,
            "lon": fallback.lon,
            "geocoder": "nominatim",
            "mb_code": fallback.mb_code,
        }
    return {"lat": pd.NA, "lon": pd.NA, "geocoder": pd.NA, "mb_code": pd.NA}


def resolve(
    in_path: Path,
    out_path: Path,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    user_agent: str = config.USER_AGENT,
    nominatim_factory: object | None = None,
    gnaf_factory: object | None = None,
) -> None:
    """Add lat/lon/geocoder/mb_code columns to ``stations.parquet``.

    Args:
        in_path: stations parquet to read.
        out_path: where to write — safe to set the same as ``in_path``.
        cache_dir: where the G-NAF + Nominatim caches live. Defaults to
            ``data/raw/geocode_cache/``.
        force: if True, re-geocode every row regardless of existing values.
        user_agent: HTTP User-Agent for Nominatim (per their usage policy).
        nominatim_factory, gnaf_factory: test seams. When provided,
            replace the real geocoders with the factory's return value.
    """
    cache_dir = cache_dir or (config.DATA_RAW / "geocode_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    stations = pd.read_parquet(in_path)
    stations = _ensure_geocoded_columns(stations)
    logger.info("loaded %d stations from %s", len(stations), in_path)

    if force:
        to_resolve = stations.index
    else:
        to_resolve = stations.index[
            stations["lat"].isna() | stations["lon"].isna()
        ]
    logger.info("resolving %d / %d station addresses", len(to_resolve), len(stations))

    if len(to_resolve) == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stations.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        return

    gnaf, nominatim = _build_geocoders(
        cache_dir, user_agent, nominatim_factory=nominatim_factory, gnaf_factory=gnaf_factory
    )

    gnaf_hits = 0
    nominatim_hits = 0
    failures = 0
    for idx in to_resolve:
        row = stations.loc[idx]
        address = _format_address(row)
        if not address:
            failures += 1
            continue
        update = _geocode_one(address, gnaf, nominatim)
        for col, val in update.items():
            stations.at[idx, col] = val
        provider = update["geocoder"]
        if pd.isna(provider):
            failures += 1
        elif provider == "gnaf":
            gnaf_hits += 1
        elif provider == "nominatim":
            nominatim_hits += 1
        else:
            failures += 1

    logger.info(
        "geocoding done: gnaf=%d nominatim=%d failures=%d",
        gnaf_hits,
        nominatim_hits,
        failures,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    stations.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    tmp.replace(out_path)
    logger.info("wrote %d stations to %s", len(stations), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    resolve(args.in_path, args.out, cache_dir=args.cache_dir, force=args.force)


if __name__ == "__main__":
    main()
