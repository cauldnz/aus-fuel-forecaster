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
import re
import time
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

# Progress-log cadence inside the per-station resolve loop. Hybrid trigger:
# whichever fires first. Rationale:
# - Time interval (10s): matches the typical "is this hung?" perception
#   threshold. Anything slower than ~30s with no output and a human
#   watching starts wondering. 10s gives steady reassurance without
#   spamming the log.
# - Count interval (250 stations): a 5,000-station NSW run on a warm
#   cache can blast through hundreds per second; the time-based trigger
#   would never fire. Count-based ensures fast runs still log
#   meaningful checkpoints (~20 messages over a 5,000-row corpus).
PROGRESS_LOG_INTERVAL_SECONDS: float = 10.0
PROGRESS_LOG_INTERVAL_COUNT: int = 250

# G-NAF parquet is published to s3://minus34.com/opendata/ (ap-southeast-2).
# The default virtual-hosted URL `minus34.com.s3.amazonaws.com` triggers a
# certificate hostname mismatch (S3's wildcard cert is `*.s3.amazonaws.com`,
# not `*.s3.amazonaws.com` *plus* a leading `.com.`). Forcing path-style via
# the regional endpoint sidesteps this and is faster (no cross-region hop).
GNAF_S3_HTTPS_ENDPOINT: str = "https://s3.ap-southeast-2.amazonaws.com"


_CROSS_STREET_RE: re.Pattern[str] = re.compile(
    r"\s+(cnr|corner|c/o)\b.*?(?=,|$)", flags=re.IGNORECASE
)

# Detect a 4-digit AU postcode at the end of a string (with or without
# a state code in front). Used to decide whether to append suburb +
# postcode or trust what's already in the address field.
_TRAILING_POSTCODE_RE: re.Pattern[str] = re.compile(r"\b\d{4}\s*$")


def _clean_street(street: str) -> str:
    """Strip cross-street annotations that break geocoder parsing.

    FuelCheck addresses often include `Cnr Ross St`, `Corner of X & Y`, etc.
    These are useful for human navigation but cause Nominatim to return zero
    hits where the same address without the annotation resolves cleanly.
    The match stops at a comma so suburb/postcode that follows is preserved.
    """
    return _CROSS_STREET_RE.sub("", street).strip()


def _normalise_postcode(raw: object) -> str:
    """Coerce a postcode value to its canonical 4-digit string form.

    `clean.fuelcheck` sometimes hands us postcodes that pandas typed as
    floats during chunked CSV reads, so a perfectly normal `2776` ends
    up serialised as the string `'2776.0'`. Strip the spurious `.0` so
    downstream G-NAF / Nominatim queries match real AU postcodes.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def _format_address(row: pd.Series) -> str:
    """Build a single address string from FuelCheck columns.

    For ~99.9% of FuelCheck stations the `address` column already has
    the canonical "<street>, <suburb> <state> <postcode>" form that the
    augmentor's `parse_address()` handles cleanly — extracting locality,
    state, and postcode into the components G-NAF Tier 2/3 need to
    resolve. Sending anything more decorated (e.g. with `, Australia`
    appended, or a duplicate postcode tacked on) corrupts the parser:
    the trailing tokens get swept into `locality` and the parser leaves
    `postcode=None` / `state=None`, killing every G-NAF tier and
    forcing every address through the rate-limited Nominatim fallback.

    Strategy:
    - Strip cross-street annotations (Nominatim still benefits from this
      even when G-NAF gets a clean exact match).
    - If the address already ends with a 4-digit postcode → return as-is.
      Don't append anything; the parser will handle it.
    - Only when the address is street-only (rare — about 1 in 6,000 NSW
      stations) do we append suburb + postcode. We never append
      "Australia": both G-NAF (AU-only by definition) and Nominatim
      handle Australian addresses fine without the country hint when a
      state code or postcode is present.
    """
    raw = str(row.get("address", "") or "").strip()
    cleaned = _clean_street(raw)
    suburb = str(row.get("suburb", "") or "").strip()
    postcode = _normalise_postcode(row.get("postcode"))

    if not cleaned:
        # Truly empty address — synthesise from the components we have.
        parts = [p for p in (suburb, postcode) if p]
        return ", ".join(parts)

    if _TRAILING_POSTCODE_RE.search(cleaned):
        # Already canonical — don't touch it. (Common case.)
        return cleaned

    # Street-only fallback: append suburb and postcode if not already
    # present. Avoid the `, Australia` suffix that breaks the parser.
    parts = [cleaned]
    lower = cleaned.lower()
    if suburb and suburb.lower() not in lower:
        parts.append(suburb)
    if postcode and postcode not in cleaned:
        parts.append(postcode)
    return ", ".join(parts)


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
            s3_https_endpoint=GNAF_S3_HTTPS_ENDPOINT,
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


def _geocode_one(
    address: str, gnaf: object | None, nominatim: object
) -> dict[str, Any]:
    """Try G-NAF first, fall back to Nominatim. Returns row-update dict.

    ``gnaf=None`` means G-NAF is disabled for this run (e.g. the remote
    parquet view failed to initialise) and every address goes to Nominatim.
    """
    if gnaf is not None:
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


def _format_eta(seconds: float) -> str:
    """Render a remaining-seconds duration as ``Hh Mm Ss`` (skip leading zeros)."""
    if seconds <= 0 or seconds != seconds:  # NaN check
        return "?"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


class _ProgressLogger:
    """Hybrid time + count progress logger for the resolve loop.

    ``maybe_emit()`` is called after every iteration; it logs only when
    either ``PROGRESS_LOG_INTERVAL_COUNT`` items have elapsed since the
    last log OR ``PROGRESS_LOG_INTERVAL_SECONDS`` have passed (whichever
    comes first). Cheap when not emitting (two comparisons + a
    ``time.monotonic()`` call), so safe to call inside the hot loop.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.start = time.monotonic()
        self.last_log_time = self.start
        self.last_log_count = 0

    def maybe_emit(
        self, processed: int, gnaf_hits: int, nominatim_hits: int, failures: int
    ) -> None:
        now = time.monotonic()
        if (
            processed - self.last_log_count < PROGRESS_LOG_INTERVAL_COUNT
            and now - self.last_log_time < PROGRESS_LOG_INTERVAL_SECONDS
        ):
            return
        elapsed = now - self.start
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = self.total - processed
        eta = _format_eta(remaining / rate) if rate > 0 else "?"
        logger.info(
            "geocoding progress: %d/%d (%.1f%%) — gnaf=%d nominatim=%d fail=%d — "
            "%.1f addr/s — eta %s",
            processed,
            self.total,
            100 * processed / self.total if self.total else 0.0,
            gnaf_hits,
            nominatim_hits,
            failures,
            rate,
            eta,
        )
        self.last_log_time = now
        self.last_log_count = processed


def _try_gnaf_warmup(gnaf: object) -> object | None:
    """Force G-NAF to open its remote view; return None if it can't.

    The augmentor opens the DuckDB connection lazily on first ``geocode()``,
    so a misconfigured remote bucket only blows up mid-loop. Probing here
    surfaces the failure early and lets the caller fall through to a
    Nominatim-only run with a clear warning.
    """
    try:
        # A nonsense address still forces the connection to open and the
        # parquet view's schema to be validated.
        gnaf.geocode("__warmup__")  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning(
            "G-NAF init failed (%s: %s) — running Nominatim-only for this session. "
            "See https://github.com/cauldnz/abs-census-augmentor/issues/8 for upstream tracking.",
            type(exc).__name__,
            exc,
        )
        return None
    return gnaf


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

    # Probe G-NAF up-front so a broken remote view fails fast — and the run
    # can degrade to Nominatim-only rather than dying mid-loop. We skip the
    # probe when a test factory injected its own G-NAF stub.
    if gnaf_factory is None:
        gnaf = _try_gnaf_warmup(gnaf)

    gnaf_hits = 0
    nominatim_hits = 0
    failures = 0
    progress = _ProgressLogger(total=len(to_resolve))
    for processed, idx in enumerate(to_resolve, start=1):
        row = stations.loc[idx]
        address = _format_address(row)
        if not address:
            failures += 1
            progress.maybe_emit(processed, gnaf_hits, nominatim_hits, failures)
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
        progress.maybe_emit(processed, gnaf_hits, nominatim_hits, failures)

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
