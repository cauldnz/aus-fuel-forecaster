"""Strict-free NOAA GFS / GEFS forecast fetcher via S3 byte-range subsetting.

Strict-free replacement for the Open-Meteo path (`fetch.weather`) — see
docs/research/2026-06_nwp_archive_alternative.md for the full design.
Session 1 scaffolding: URL routing across 3 GEFS/GFS path schemes, `.idx`
parsing, byte-range GRIB fetch, and a single-variable end-to-end smoke
function. Session 2 (this module's `fetch_one_day_all_horizons` +
per-(date, horizon) grid parquet writer + CLI + parallel orchestrator)
builds on top.

Three S3 path schemes (empirically confirmed in research doc):

1. GEFS 1° **pre-2019** (2017-01-01 .. 2018-12-31)
   - bucket  : noaa-gefs-pds
   - layout  : gefs.YYYYMMDD/HH/gec00.t{HH}z.pgrb2af{NNN}
   - control member at top level, 3-digit lead time (zero-padded).

2. GEFS 1° **2019..early-2020** (2019-01-01 .. 2020-09-22)
   - bucket  : noaa-gefs-pds
   - layout  : gefs.YYYYMMDD/HH/pgrb2a/gec00.t{HH}z.pgrb2af{NN}
   - control member in `pgrb2a` subdir, 2-digit lead time (NB: `pgrb2af24`,
     not `pgrb2af024`).

3. GEFSv12 0.5° (2020-09-23 .. 2021-03-31, used as bridge)
   - bucket  : noaa-gefs-pds
   - layout  : gefs.YYYYMMDD/HH/atmos/pgrb2ap5/geavg.t{HH}z.pgrb2a.0p50.f{NNN}
   - ensemble mean in `atmos/pgrb2ap5` subdir, 3-digit lead time.

4. GFS 0.25° (2021-04-01 onward)
   - bucket  : noaa-gfs-bdp-pds
   - layout  : gfs.YYYYMMDD/HH/atmos/gfs.t{HH}z.pgrb2.0p25.f{NNN}
   - deterministic GFS, 3-digit lead time.

A date-based selector (`_select_resolution_for_date`) routes between the
four. Year-end transitions where the layout *or* the resolution flips are
covered by tests at the boundary dates.

The byte-range fetch pattern:
  1. GET `<file_url>.idx` (~3-50 KB text), parse to {var_key -> (off, off_end)}.
  2. For each desired variable, issue HTTP Range request on the GRIB file.
  3. Concatenate mini-GRIBs (each starts with `GRIB` and ends with `7777`)
     into a single byte buffer.
  4. Write to temp file (cfgrib's stream support is limited) and parse via
     xarray + cfgrib engine.

Spec: spec.md §13.7, §13.8.
Research: docs/research/2026-06_nwp_archive_alternative.md.
"""
from __future__ import annotations

import datetime as dt
import logging
import tempfile
from pathlib import Path
from typing import Final, Literal

import requests
import xarray as xr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from fuel_pred import config

logger = logging.getLogger(__name__)

# Resolution / pipeline-window literal. Used by `_select_resolution_for_date`.
ResolutionKey = Literal["gfs_0p25", "gefs_0p5", "gefs_1"]

# S3 bucket roots — public, anonymous HTTPS.
ARCHIVE_URLS: Final[dict[str, str]] = {
    "gfs_bucket":  "https://noaa-gfs-bdp-pds.s3.amazonaws.com",
    "gefs_bucket": "https://noaa-gefs-pds.s3.amazonaws.com",
}

# Date-window dispatch (research doc §"Recommended approach"). Boundaries
# are inclusive. Pre-2017 dates fall through to Open-Meteo Archive (not
# this fetcher's job; see Session 4 router).
GEFS_1DEG_START: Final[dt.date] = dt.date(2017, 1, 1)
# Last day of GEFS pre-v12 1° (control member). 2019-01-01 introduces the
# `pgrb2a` subdir but the resolution stays at 1° and the variable inventory
# is the same — handled by the URL builder.
GEFS_1DEG_END: Final[dt.date] = dt.date(2020, 9, 22)
# GEFSv12 0.5° ensemble mean bridge.
GEFS_05DEG_START: Final[dt.date] = dt.date(2020, 9, 23)
GEFS_05DEG_END: Final[dt.date] = dt.date(2021, 3, 31)
# GFS 0.25° deterministic, strict start.
GFS_025DEG_START: Final[dt.date] = dt.date(2021, 4, 1)

# Layout-2 boundary (within the 1° window): pgrb2a subdir + 2-digit lead.
GEFS_LAYOUT2_START: Final[dt.date] = dt.date(2019, 1, 1)


def _select_resolution_for_date(date: dt.date) -> ResolutionKey:
    """Pick the per-date resolution per the hybrid table (research §"Recommended approach").

    Raises ValueError for pre-2017-01-01 dates — those route to Open-Meteo
    Archive ERA5 (handled by the Session-4 router, not this module).
    """
    if date < GEFS_1DEG_START:
        raise ValueError(
            f"date {date} is before GFS/GEFS coverage start ({GEFS_1DEG_START}); "
            "use Open-Meteo Archive (ERA5) for pre-2017 dates",
        )
    if date <= GEFS_1DEG_END:
        return "gefs_1"
    if date <= GEFS_05DEG_END:
        return "gefs_0p5"
    return "gfs_0p25"


def _build_url(date: dt.date, cycle: str, lead_h: int, resolution: ResolutionKey) -> str:
    """Construct the GRIB file URL for a (date, cycle, lead-hour, resolution) triple.

    Args:
        date: forecast initialization date (UTC).
        cycle: forecast cycle as 2-digit hour string. "00", "06", "12", "18".
        lead_h: forecast lead time in hours (e.g. 24 for day-1, 168 for day-7).
        resolution: one of "gfs_0p25", "gefs_0p5", "gefs_1" — caller must
            already have routed via :func:`_select_resolution_for_date` or
            equivalent.

    Returns the full HTTPS URL of the GRIB file (no `.idx` suffix).
    """
    if cycle not in {"00", "06", "12", "18"}:
        raise ValueError(f"cycle must be one of 00/06/12/18; got {cycle!r}")
    if lead_h < 0:
        raise ValueError(f"lead_h must be non-negative; got {lead_h}")

    ymd = date.strftime("%Y%m%d")

    if resolution == "gfs_0p25":
        # GFS 0.25° — present layout. 3-digit lead, e.g. f024, f168.
        return (
            f"{ARCHIVE_URLS['gfs_bucket']}/"
            f"gfs.{ymd}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{lead_h:03d}"
        )

    if resolution == "gefs_0p5":
        # GEFSv12 0.5° ensemble mean.
        return (
            f"{ARCHIVE_URLS['gefs_bucket']}/"
            f"gefs.{ymd}/{cycle}/atmos/pgrb2ap5/"
            f"geavg.t{cycle}z.pgrb2a.0p50.f{lead_h:03d}"
        )

    # resolution == "gefs_1" — two sub-layouts at the 2019-01-01 boundary.
    if date < GEFS_LAYOUT2_START:
        # Pre-2019 layout: 3-digit lead at top level.
        return (
            f"{ARCHIVE_URLS['gefs_bucket']}/"
            f"gefs.{ymd}/{cycle}/gec00.t{cycle}z.pgrb2af{lead_h:03d}"
        )
    # 2019-01-01 .. 2020-09-22 layout: pgrb2a subdir, 2-digit lead.
    return (
        f"{ARCHIVE_URLS['gefs_bucket']}/"
        f"gefs.{ymd}/{cycle}/pgrb2a/gec00.t{cycle}z.pgrb2af{lead_h:02d}"
    )


# ------------------------------ HTTP layer ------------------------------


def _http_get(url: str, *, headers: dict[str, str] | None = None) -> requests.Response:
    """Plain HTTPS GET with project User-Agent. Raises for HTTP errors."""
    merged: dict[str, str] = {"User-Agent": config.USER_AGENT}
    if headers:
        merged.update(headers)
    r = requests.get(url, headers=merged, timeout=config.REQUEST_TIMEOUT)
    r.raise_for_status()
    return r


@retry(
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=config.RETRY_BACKOFF_SECONDS, max=30),
    retry=retry_if_exception_type((requests.RequestException, OSError)),
    reraise=True,
)
def _fetch_idx(url: str) -> dict[str, tuple[int, int]]:
    """Fetch and parse a GFS/GEFS `.idx` sidecar.

    Returns a dict keyed by ``"{short_name}:{level}"`` (e.g. ``"TMAX:2 m above ground"``)
    mapping to ``(start_byte, end_byte)`` half-open byte range suitable for
    an HTTP Range header. ``end_byte`` is exclusive — use ``end_byte - 1``
    in ``bytes=start-(end-1)``.

    Note: ``.idx`` lines have the form
        ``record_no:byte_offset:date:variable:level:fcst_type[:ensemble]``
    Some files contain multiple records with the same `variable:level`
    combination (different `fcst_type`, e.g. 0-3 vs 0-6 hour acc) — this
    parser keeps the **last** match per key. Callers that need
    fcst-type-specific records (e.g. f024 vs f006 max blocks) should
    construct keys with the fcst suffix themselves (see :func:`_fetch_idx_full`).
    """
    text = _http_get(url).text
    out: dict[str, tuple[int, int]] = {}
    rows = []
    for raw in text.strip().split("\n"):
        parts = raw.split(":")
        if len(parts) < 6:
            continue
        rec_no = int(parts[0])
        offset = int(parts[1])
        var = parts[3]
        level = parts[4]
        rows.append((rec_no, offset, var, level))

    # Sort by record number to compute end-offsets robustly.
    rows.sort(key=lambda t: t[0])
    for i, (_rec, off, var, level) in enumerate(rows):
        next_off = rows[i + 1][1] if i + 1 < len(rows) else off  # last record has no next
        key = f"{var}:{level}"
        # If a later record overrides, keep the later (matches the doc's
        # "last match wins" — but in practice unique-keyed records dominate).
        out[key] = (off, next_off)
    return out


@retry(
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=config.RETRY_BACKOFF_SECONDS, max=30),
    retry=retry_if_exception_type((requests.RequestException, OSError)),
    reraise=True,
)
def _fetch_records(url: str, byte_ranges: list[tuple[int, int]]) -> bytes:
    """Fetch one-or-more byte ranges from `url`, return concatenated bytes.

    Each range is ``(start, end_exclusive)``; an HTTP 206 Partial Content
    is expected per request. Mini-GRIBs are concatenable as-is because each
    starts with the `GRIB` magic and ends with `7777`.

    For a single byte range, prefer one HTTP call; for many, this issues
    them serially. (S3 supports multi-range requests in theory but is
    inconsistent in practice; serial is simpler and the overhead is small.)
    """
    if not byte_ranges:
        return b""
    chunks: list[bytes] = []
    for start, end_exclusive in byte_ranges:
        # end_exclusive is the offset of the *next* record (or the
        # file-end-marker for the last). Range header is inclusive on both
        # ends; subtract 1 to get the last byte of the record.
        r = _http_get(url, headers={"Range": f"bytes={start}-{end_exclusive - 1}"})
        if r.status_code not in (200, 206):
            raise RuntimeError(
                f"unexpected HTTP {r.status_code} for byte range from {url}: "
                f"expected 206 Partial Content",
            )
        chunks.append(r.content)
    return b"".join(chunks)


def _parse_grib_to_xarray(grib_bytes: bytes, var_key: str) -> xr.DataArray:
    """Parse a (possibly multi-record) GRIB byte buffer to one xarray DataArray.

    cfgrib's xarray engine requires a file path (its stream support is
    limited and unreliable for mini-GRIBs). We write to a temp file in the
    OS temp dir, parse, then unlink.

    Args:
        grib_bytes: raw GRIB2 bytes; must start with `GRIB` magic.
        var_key: variable key the caller is looking for, format
            ``"{SHORT_NAME}:{level}"`` e.g. ``"TMAX:2 m above ground"``.
            cfgrib renames short names to lower-case (e.g. ``TMAX`` -> ``tmax``);
            if the dataset only has one variable (typical for single-record
            mini-GRIBs) we just return that variable without name-matching.

    Returns:
        The xarray DataArray for the matched variable. Dimensions typically
        include ``latitude``, ``longitude`` (and ``step``/``time`` as scalar
        coords).
    """
    if not grib_bytes.startswith(b"GRIB"):
        raise RuntimeError(
            f"GRIB byte buffer doesn't start with magic 'GRIB' "
            f"(got {grib_bytes[:8]!r}, len={len(grib_bytes)})",
        )

    # cfgrib needs a real file path. Use a temp file in the OS temp dir.
    # `delete=False` because Windows won't reopen a still-open NamedTemporaryFile;
    # we manage the unlink ourselves in `finally`.
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        tmp_path = Path(f.name)
    try:
        ds = xr.open_dataset(
            tmp_path,
            engine="cfgrib",
            # Tell cfgrib not to write a .idx alongside (clutters tmp).
            backend_kwargs={"indexpath": ""},
        )
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise RuntimeError(f"no data variables in GRIB; key was {var_key!r}")
        # When the buffer has one record, there's exactly one variable.
        # When it has multiple records of the same `step`, cfgrib coalesces
        # them; multiple `step`s usually error unless we filter — single-record
        # is the common case for Session 1's smoke path.
        var_name = data_vars[0]
        da = ds[var_name]
        # Ensure cfgrib's lazy data is materialised before we delete the
        # tmp file. xarray's `.load()` reads everything into memory.
        return da.load()
    finally:
        try:
            tmp_path.unlink()
        except OSError as exc:  # pragma: no cover — defensive
            logger.debug("could not unlink tmp grib %s: %s", tmp_path, exc)


# ------------------------------ Smoke test ------------------------------


def fetch_one_variable_one_date(
    var_key: str,
    date: dt.date,
    cycle: str,
    lead_h: int,
    resolution: ResolutionKey | None = None,
) -> xr.DataArray:
    """End-to-end smoke fetch: one variable, one date, one cycle, one lead-time.

    Used to verify the byte-range / cfgrib pipeline against live S3. For
    production fetches see :func:`fetch_one_day_all_horizons` (Session 2).

    Args:
        var_key: GRIB key, format ``"{SHORT_NAME}:{level}"`` —
            e.g. ``"TMAX:2 m above ground"``, ``"APCP:surface"``.
        date: forecast init date.
        cycle: forecast cycle hour, e.g. ``"00"``.
        lead_h: forecast lead in hours (e.g. ``24`` for day-1).
        resolution: explicit resolution override; defaults to
            :func:`_select_resolution_for_date`.

    Returns:
        xarray DataArray of the variable's global field.
    """
    res: ResolutionKey = resolution or _select_resolution_for_date(date)
    grib_url = _build_url(date, cycle, lead_h, res)
    idx_url = grib_url + ".idx"
    logger.info("fetch.gfs smoke: %s @ %s (lead=%dh, %s)", var_key, date, lead_h, res)

    idx = _fetch_idx(idx_url)
    if var_key not in idx:
        raise KeyError(
            f"variable {var_key!r} not in idx for {grib_url} — "
            f"available keys (first 5): {list(idx)[:5]}",
        )
    start, end_exclusive = idx[var_key]
    logger.debug("byte range: %d-%d (size=%d)", start, end_exclusive - 1, end_exclusive - start)

    grib_bytes = _fetch_records(grib_url, [(start, end_exclusive)])
    return _parse_grib_to_xarray(grib_bytes, var_key)


__all__ = [
    "ARCHIVE_URLS",
    "GEFS_05DEG_END",
    "GEFS_05DEG_START",
    "GEFS_1DEG_END",
    "GEFS_1DEG_START",
    "GEFS_LAYOUT2_START",
    "GFS_025DEG_START",
    "ResolutionKey",
    "_build_url",
    "_fetch_idx",
    "_fetch_records",
    "_parse_grib_to_xarray",
    "_select_resolution_for_date",
    "fetch_one_variable_one_date",
]
