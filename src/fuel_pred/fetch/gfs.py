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

import argparse
import datetime as dt
import logging
import tempfile
from pathlib import Path
from typing import Final, Literal

import cfgrib
import numpy as np
import pandas as pd
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


# ------------------------- Session 2: multi-horizon + grid parquet -------------------------

# NSW bounding box (degrees) — generous of NSW state boundaries to include
# a one-cell halo for bilinear interp at the boundary. Lat in [-37.5, -28],
# lon in [140.5, 154].
NSW_LAT_MIN: Final[float] = -37.5
NSW_LAT_MAX: Final[float] = -28.0
NSW_LON_MIN: Final[float] = 140.5
NSW_LON_MAX: Final[float] = 154.0

# Default forecast horizons (days). Day-1..Day-7 = the v2.1 multi-horizon
# scope. Caller can subset via the CLI `--horizons` flag.
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)
DEFAULT_CYCLE: Final[str] = "00"

# Variable mapping. Each entry: (grib_idx_key, our_column_name, reducer).
# The idx key is what `_fetch_idx` returns as the dict key — format is
# `"{SHORT_NAME}:{level}"`. `reducer` is the per-(grid_cell, variable)
# aggregation across the 4 6h leads making up the horizon's day:
#   "max" — TMAX block max across leads (Kelvin)
#   "min" — TMIN block min across leads (Kelvin)
#   "sum" — APCP accumulation summed across leads (mm)
#   "wind_max" — special-case: requires UGRD+VGRD pair, computes
#                sqrt(U^2 + V^2) per lead, then max across leads (m/s)
# Temperature columns are converted K→°C and wind m/s→km/h after aggregation.
GRIB_VARIABLES: Final[dict[str, tuple[str, str]]] = {
    "wx_temp_max_c": ("TMAX:2 m above ground", "max"),
    "wx_temp_min_c": ("TMIN:2 m above ground", "min"),
    "wx_precipitation_mm": ("APCP:surface", "sum"),
    # Wind is a U/V composite; see _aggregate_wind_speed_max.
    # The "key" here is unused for wind — special-cased.
    "wx_wind_speed_max_kmh": ("UGRD:10 m above ground+VGRD:10 m above ground", "wind_max"),
}
# Output schema. wx_weather_code is stubbed null per research doc R3 (GFS
# doesn't emit WMO codes; v1 SHAP shows the column has low importance —
# revisit derivation later if A/B flagged).
OUTPUT_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "wx_temp_max_c",
    "wx_temp_min_c",
    "wx_precipitation_mm",
    "wx_wind_speed_max_kmh",
)


def _leads_for_horizon(horizon: int) -> list[int]:
    """Return the 4 6h lead-hour offsets that aggregate to one daily window for horizon N.

    Horizon 1 → [6, 12, 18, 24] (covers UTC day 0-24h of forecast).
    Horizon 2 → [30, 36, 42, 48].
    Horizon 7 → [150, 156, 162, 168].

    These align to UTC-day boundaries (research doc §"Day boundary alignment").
    The aggregation is a UTC-day max/min/sum, not Sydney-local-day — see
    docs/research/2026-06_nwp_archive_alternative.md R2 for the decision.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    base = 24 * (horizon - 1)
    return [base + 6, base + 12, base + 18, base + 24]


def _open_grib_dataset(grib_bytes: bytes) -> xr.Dataset:
    """Parse a (possibly multi-record) GRIB byte buffer to an xarray Dataset.

    Unlike :func:`_parse_grib_to_xarray` this handles mixed-type concatenations
    (e.g. TMAX 18-24hr-max + APCP 18-24hr-acc + U10 instantaneous in one
    buffer) by going through ``cfgrib.open_datasets`` and merging the result
    Datasets. The merge is safe because each variable has a unique name in
    the returned Datasets (`tmax`, `tmin`, `tp`, `u10`, `v10`) and they all
    share the same lat/lon dims after cfgrib's normalisation.

    Returns the merged Dataset with all data loaded into memory (the
    underlying temp file is deleted before return).
    """
    if not grib_bytes.startswith(b"GRIB"):
        raise RuntimeError(
            f"GRIB byte buffer doesn't start with magic 'GRIB' "
            f"(got {grib_bytes[:8]!r}, len={len(grib_bytes)})",
        )
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        tmp_path = Path(f.name)
    try:
        # cfgrib.open_datasets returns a list of Datasets, one per
        # cfgrib "filter" group (records that share dimension structure
        # — e.g. heightAboveGround=2 for TMAX/TMIN, surface for APCP,
        # heightAboveGround=10 for U10/V10). We merge them — variable
        # names are unique across the groups so no collision.
        dsets = cfgrib.open_datasets(
            str(tmp_path),
            backend_kwargs={"indexpath": ""},
        )
        if not dsets:
            raise RuntimeError("cfgrib returned no datasets")
        # Drop coords that may conflict on merge (each dataset has its
        # own scalar `heightAboveGround` / `surface` / `step` etc.).
        # Keep only lat/lon as the merge axes.
        cleaned: list[xr.Dataset] = []
        for d in dsets:
            keep_coords = {c for c in d.coords if c in ("latitude", "longitude")}
            drop = [c for c in d.coords if c not in keep_coords]
            cleaned.append(d.reset_coords(drop, drop=True))
        merged = xr.merge(cleaned, compat="override")
        return merged.load()
    finally:
        try:
            tmp_path.unlink()
        except OSError as exc:  # pragma: no cover — defensive
            logger.debug("could not unlink tmp grib %s: %s", tmp_path, exc)


def _select_nsw_box(da: xr.DataArray) -> xr.DataArray:
    """Subset a global GFS/GEFS field to the NSW bounding box.

    Handles cfgrib's lat-descending convention via `sel(latitude=slice(...))`
    by passing the slice in descending order. Lon is ascending 0..360 and
    NSW lons (140..154) sit in the positive-only range — no wrap.

    Returns a copy with `latitude` and `longitude` coords reset so the index
    values match the global grid (downstream join uses global indices from
    the station grid mapping).
    """
    return da.sel(
        latitude=slice(NSW_LAT_MAX, NSW_LAT_MIN),  # descending — top to bottom
        longitude=slice(NSW_LON_MIN, NSW_LON_MAX),
    )


def _aggregate_per_variable(
    per_lead_arrays: dict[int, xr.DataArray], reducer: str,
) -> xr.DataArray:
    """Reduce 4 per-lead arrays into one daily aggregate (max/min/sum)."""
    if not per_lead_arrays:
        raise ValueError("per_lead_arrays is empty")
    leads_sorted = sorted(per_lead_arrays)
    stacked = xr.concat(
        [per_lead_arrays[lh] for lh in leads_sorted],
        dim="lead",
    )
    if reducer == "max":
        return stacked.max(dim="lead")
    if reducer == "min":
        return stacked.min(dim="lead")
    if reducer == "sum":
        return stacked.sum(dim="lead")
    raise ValueError(f"unknown reducer {reducer!r}")


def _aggregate_wind_speed_max(
    u_per_lead: dict[int, xr.DataArray], v_per_lead: dict[int, xr.DataArray],
) -> xr.DataArray:
    """Compute scalar wind speed `sqrt(U² + V²)` per lead, then max across leads.

    Returns a DataArray in m/s — caller converts to km/h.

    Per research doc R4: this is "instantaneous max-of-4-snapshots", not
    true daily wind max. GUST:surface would be a better proxy, but it's
    not consistently available across all 3 layouts (research doc says
    GFS has it; GEFS pre-v12 has it on some records). For first ship, the
    U/V composite gives a consistent definition across all three sources.
    """
    if set(u_per_lead) != set(v_per_lead):
        raise ValueError(
            f"U/V leads mismatch: U={sorted(u_per_lead)} V={sorted(v_per_lead)}",
        )
    speeds: dict[int, xr.DataArray] = {}
    for lh in u_per_lead:
        # `**0.5` on an xr.DataArray returns a DataArray (xarray's pow
        # dunder). Avoid `np.sqrt(...)` here because mypy widens its
        # return type to `ndarray`.
        speeds[lh] = (u_per_lead[lh] ** 2 + v_per_lead[lh] ** 2) ** 0.5
    return _aggregate_per_variable(speeds, "max")


def _fetch_lead(
    date: dt.date, cycle: str, lead_h: int, resolution: ResolutionKey,
) -> xr.Dataset:
    """Fetch the 5 spec'd variables for one (date, cycle, lead_h) from GFS/GEFS S3.

    Returns the merged xarray Dataset containing TMAX, TMIN, APCP, U10, V10.
    Each variable is a global (lat, lon) field. Caller is responsible for
    NSW subset + aggregation.

    Raises:
        KeyError if any required variable is missing from the .idx (rare;
        empirically all 5 are present in every cycle/lead within the 3
        layout windows in scope).
    """
    grib_url = _build_url(date, cycle, lead_h, resolution)
    idx_url = grib_url + ".idx"
    idx = _fetch_idx(idx_url)
    # Required variable keys (one per logical column; wind is U+V).
    needed_keys = [
        "TMAX:2 m above ground",
        "TMIN:2 m above ground",
        "APCP:surface",
        "UGRD:10 m above ground",
        "VGRD:10 m above ground",
    ]
    byte_ranges: list[tuple[int, int]] = []
    for k in needed_keys:
        if k not in idx:
            raise KeyError(
                f"variable {k!r} missing from {idx_url} — available (first 10): "
                f"{list(idx)[:10]}",
            )
        byte_ranges.append(idx[k])
    grib_bytes = _fetch_records(grib_url, byte_ranges)
    return _open_grib_dataset(grib_bytes)


def fetch_one_day_all_horizons(
    date: dt.date,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    cycle: str = DEFAULT_CYCLE,
) -> dict[int, pd.DataFrame]:
    """Fetch & aggregate one init-date across all requested horizons.

    Returns a dict keyed by horizon (1..7), each value a DataFrame with
    columns `lat_idx, lon_idx, wx_temp_max_c, wx_temp_min_c,
    wx_precipitation_mm, wx_wind_speed_max_kmh` covering the NSW box at
    the resolution selected for `date`.

    `wx_weather_code` is intentionally absent — write_grid_parquet adds it
    as a null column to match the spec output schema.

    Args:
        date: forecast init date (UTC). Routed to GFS/GEFS resolution
            automatically via :func:`_select_resolution_for_date`.
        horizons: forecast-day horizons to compute. Each horizon needs
            4 successive 6h leads from the init time.
        cycle: forecast cycle hour, default "00" (00Z run; matches the
            project's single-cycle/day backfill plan).

    Note: the lat_idx/lon_idx are global grid indices (matching the indices
    in `station_grid_mapping.parquet`), not NSW-box-local indices. This
    keeps the Session-3 feature-build join one-step.
    """
    resolution = _select_resolution_for_date(date)
    logger.info(
        "gfs.fetch: date=%s cycle=%sZ horizons=%s resolution=%s",
        date, cycle, list(horizons), resolution,
    )

    # All unique leads we'll touch across requested horizons.
    all_leads: set[int] = set()
    for h in horizons:
        all_leads.update(_leads_for_horizon(h))

    # Pull each lead's full dataset (5 vars) once — reused across horizons
    # that share leads. (In the default config no overlap, but the structure
    # generalises.)
    per_lead_ds: dict[int, xr.Dataset] = {}
    for lh in sorted(all_leads):
        per_lead_ds[lh] = _fetch_lead(date, cycle, lh, resolution)
        logger.debug("fetched lead %s for %s @ %sZ", lh, date, cycle)

    # cfgrib variable-name conventions in the returned Datasets:
    # GFS 0.25°  : tmax, tmin, tp (total precipitation), u10, v10
    # GEFS 1°    : tmax, tmin, tp, u10, v10  — same names
    # GEFS 0.5°  : tmax, tmin, tp, u10, v10  — same names
    # We retrieve by xarray variable name (after cfgrib's rename), not by
    # GRIB shortName. Helper that's tolerant to slight naming variation:
    def _pick(ds: xr.Dataset, candidates: tuple[str, ...]) -> xr.DataArray:
        for name in candidates:
            if name in ds.data_vars:
                return ds[name]
        raise KeyError(
            f"none of {candidates} in dataset vars {list(ds.data_vars)}",
        )

    # Build per-horizon aggregates.
    out: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        leads_h = _leads_for_horizon(horizon)

        tmax_per_lead = {lh: _pick(per_lead_ds[lh], ("tmax",)) for lh in leads_h}
        tmin_per_lead = {lh: _pick(per_lead_ds[lh], ("tmin",)) for lh in leads_h}
        # APCP/total-precip variant naming: cfgrib calls it `tp` for newer
        # data, `tp` for GEFSv12; the GRIB shortName is `tp`. Older GEFS
        # may use `acpcp`. Be defensive.
        apcp_per_lead = {lh: _pick(per_lead_ds[lh], ("tp", "acpcp")) for lh in leads_h}
        u_per_lead = {lh: _pick(per_lead_ds[lh], ("u10",)) for lh in leads_h}
        v_per_lead = {lh: _pick(per_lead_ds[lh], ("v10",)) for lh in leads_h}

        tmax_agg = _aggregate_per_variable(tmax_per_lead, "max")
        tmin_agg = _aggregate_per_variable(tmin_per_lead, "min")
        apcp_agg = _aggregate_per_variable(apcp_per_lead, "sum")
        wind_agg = _aggregate_wind_speed_max(u_per_lead, v_per_lead)

        # NSW subset + unit conversions.
        tmax_nsw = _select_nsw_box(tmax_agg) - 273.15  # K → °C
        tmin_nsw = _select_nsw_box(tmin_agg) - 273.15
        apcp_nsw = _select_nsw_box(apcp_agg)           # already in mm
        wind_nsw = _select_nsw_box(wind_agg) * 3.6     # m/s → km/h

        out[horizon] = _stack_to_grid_frame(tmax_nsw, tmin_nsw, apcp_nsw, wind_nsw)

    return out


def _stack_to_grid_frame(
    tmax: xr.DataArray, tmin: xr.DataArray,
    apcp: xr.DataArray, wind: xr.DataArray,
) -> pd.DataFrame:
    """Stack 4 NSW-box DataArrays into one long-form grid DataFrame.

    Output columns: ``lat_idx, lon_idx, lat, lon, wx_temp_max_c,
    wx_temp_min_c, wx_precipitation_mm, wx_wind_speed_max_kmh``.

    `lat_idx`/`lon_idx` are GLOBAL grid indices (using the same convention
    as :func:`fuel_pred.spatial.gfs_grid._bilinear_for_station` — i.e.
    ``lat_idx = round((90 - lat) / res)``, ``lon_idx = round(lon / res)``).
    `lat`/`lon` are the original degree values (helpful for debugging /
    plotting; downstream joins are by index, not by value).

    `wx_weather_code` is intentionally omitted per research doc R3 — GFS
    doesn't emit WMO codes and v1 SHAP shows the column has low importance.
    If feature-build needs the column, it adds a null stub there.
    """
    # Infer resolution from the lat/lon spacing.
    lats = tmax.latitude.values
    lons = tmax.longitude.values
    if len(lats) < 2 or len(lons) < 2:
        raise RuntimeError("NSW subset too small to infer resolution")
    res_lat = float(abs(lats[1] - lats[0]))  # lats descend → use abs
    res_lon = float(abs(lons[1] - lons[0]))

    # Global indices match the station→grid mapping convention.
    lat_global_indices = np.round((90.0 - lats) / res_lat).astype(np.int64)
    lon_global_indices = np.round(lons / res_lon).astype(np.int64)

    n_lat = len(lats)
    n_lon = len(lons)

    # Long-form (lat × lon) grid via meshgrid + flatten.
    lat_idx_grid, lon_idx_grid = np.meshgrid(
        lat_global_indices, lon_global_indices, indexing="ij",
    )
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    df = pd.DataFrame({
        "lat_idx": lat_idx_grid.flatten(),
        "lon_idx": lon_idx_grid.flatten(),
        "lat": lat_grid.flatten().astype(np.float64),
        "lon": lon_grid.flatten().astype(np.float64),
        "wx_temp_max_c": tmax.values.flatten().astype(np.float64),
        "wx_temp_min_c": tmin.values.flatten().astype(np.float64),
        "wx_precipitation_mm": apcp.values.flatten().astype(np.float64),
        "wx_wind_speed_max_kmh": wind.values.flatten().astype(np.float64),
    })
    # Sanity: row count should equal n_lat × n_lon.
    if len(df) != n_lat * n_lon:
        raise RuntimeError(
            f"flattened grid has {len(df)} rows; expected {n_lat * n_lon}",
        )

    return df


def _grid_parquet_path(out_dir: Path, date: dt.date, horizon: int) -> Path:
    """Canonical per-(date, horizon) output path: <out_dir>/<YYYY-MM-DD>_h<N>.parquet."""
    return out_dir / f"{date.isoformat()}_h{horizon}.parquet"


def fetch_and_write_one_day(
    date: dt.date,
    out_dir: Path,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    cycle: str = DEFAULT_CYCLE,
    force: bool = False,
) -> list[Path]:
    """Fetch one day's worth of horizons; write each as atomic per-(date, horizon) parquet.

    Returns the list of paths written (or already-cached, when `force` is False).

    Cache check: if every (date, horizon) parquet already exists AND `force`
    is False, the fetch is skipped entirely (no network calls).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [_grid_parquet_path(out_dir, date, h) for h in horizons]

    if not force and all(p.exists() for p in out_paths):
        logger.debug("cache hit for %s — all %d horizons present", date, len(horizons))
        return out_paths

    horizon_frames = fetch_one_day_all_horizons(date, horizons=horizons, cycle=cycle)

    written: list[Path] = []
    for horizon, out_path in zip(horizons, out_paths, strict=True):
        df = horizon_frames[horizon]
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        df.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
        tmp_path.replace(out_path)
        logger.info(
            "wrote %d cells for %s h=%d to %s", len(df), date, horizon, out_path.name,
        )
        written.append(out_path)
    return written


def fetch(
    start: str,
    end: str,
    out_dir: Path,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    cycle: str = DEFAULT_CYCLE,
    force: bool = False,
) -> None:
    """Fetch a date range's worth of (date × horizon) grid parquets serially.

    Skips dates whose parquets are all cached (per :func:`fetch_and_write_one_day`).
    For parallel execution at scale, see `tools/parallel_gfs_fetch.py`.
    """
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    if start_d > end_d:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    dates = [start_d + dt.timedelta(days=n) for n in range((end_d - start_d).days + 1)]
    logger.info(
        "gfs.fetch range: %d dates × %d horizons = %d (date, horizon) files",
        len(dates), len(horizons), len(dates) * len(horizons),
    )

    fetched_dates = 0
    cached_dates = 0
    failed_dates = 0
    for i, d in enumerate(dates, start=1):
        # Cache check at the outer loop so we can count it. fetch_and_write_one_day
        # repeats the check but is idempotent — a few stat() calls aren't worth
        # the API split.
        expected = [_grid_parquet_path(out_dir, d, h) for h in horizons]
        if not force and all(p.exists() for p in expected):
            cached_dates += 1
            continue
        try:
            fetch_and_write_one_day(
                d, out_dir, horizons=horizons, cycle=cycle, force=force,
            )
            fetched_dates += 1
        except Exception:
            failed_dates += 1
            logger.exception("failed to fetch %s", d)

        if i % 50 == 0 or i == len(dates):
            logger.info(
                "progress: %d / %d  fetched=%d cached=%d failed=%d",
                i, len(dates), fetched_dates, cached_dates, failed_dates,
            )

    logger.info(
        "gfs.fetch complete: total_dates=%d fetched=%d cached=%d failed=%d",
        len(dates), fetched_dates, cached_dates, failed_dates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True,
                        help="ISO date (YYYY-MM-DD) — inclusive.")
    parser.add_argument("--end", required=True,
                        help="ISO date (YYYY-MM-DD) — inclusive.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output dir for per-(date, horizon) grid parquets")
    parser.add_argument(
        "--horizons", default="1,2,3,4,5,6,7",
        help="Comma-separated horizon days (1..7). Default: 1..7.",
    )
    parser.add_argument("--cycle", default=DEFAULT_CYCLE, choices=["00", "06", "12", "18"])
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch ignoring cache.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    horizons = tuple(int(x) for x in args.horizons.split(","))
    fetch(
        start=args.start, end=args.end, out_dir=args.out,
        horizons=horizons, cycle=args.cycle, force=args.force,
    )


__all__ = [
    "ARCHIVE_URLS",
    "DEFAULT_CYCLE",
    "DEFAULT_HORIZONS",
    "GEFS_05DEG_END",
    "GEFS_05DEG_START",
    "GEFS_1DEG_END",
    "GEFS_1DEG_START",
    "GEFS_LAYOUT2_START",
    "GFS_025DEG_START",
    "GRIB_VARIABLES",
    "NSW_LAT_MAX",
    "NSW_LAT_MIN",
    "NSW_LON_MAX",
    "NSW_LON_MIN",
    "OUTPUT_VALUE_COLUMNS",
    "ResolutionKey",
    "_aggregate_per_variable",
    "_aggregate_wind_speed_max",
    "_build_url",
    "_fetch_idx",
    "_fetch_lead",
    "_fetch_records",
    "_grid_parquet_path",
    "_leads_for_horizon",
    "_open_grib_dataset",
    "_parse_grib_to_xarray",
    "_select_nsw_box",
    "_select_resolution_for_date",
    "_stack_to_grid_frame",
    "fetch",
    "fetch_and_write_one_day",
    "fetch_one_day_all_horizons",
    "fetch_one_variable_one_date",
]


if __name__ == "__main__":
    main()
