"""Fetch NSW Roads Traffic Volume Counts from TfNSW open-data portal.

Source: https://opendata.transport.nsw.gov.au/data/dataset/nsw-roads-traffic-volume-counts-api
Granularity: hourly per station; clean.traffic aggregates to daily.
Coverage: 2006 → present (with documented data-quality issues).

Tables fetched (spec.md §5.1 reference):
- Road Traffic Counts Station Reference   → ``<out_dir>/stations.parquet``
- Road Traffic Counts Hourly Permanent     → ``<out_dir>/hourly.parquet``

Spec divergence:
- The spec hint described both tables as CKAN datastores paginated via
  ``datastore_search``. In reality (verified May 2026):
    * the **stations** resource is a CKAN datastore (``datastore_active=True``)
      and is fetched per the spec hint;
    * the **hourly** resource is a single ZIP download containing one or
      more CSVs — there is no datastore to paginate.
  We handle both shapes.
TODO(spec): update spec.md §5.1 to record the ZIP shape of the hourly
resource so future readers don't follow the stale hint.

Spec: spec.md §5.1.
"""
from __future__ import annotations

import argparse
import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from fuel_pred.fetch import _ckan

logger = logging.getLogger(__name__)

API_ROOT: str = "https://opendata.transport.nsw.gov.au/api/3/action"
PACKAGE_ID: str = "nsw-roads-traffic-volume-counts-api"

# Substring match (case-insensitive) on the CKAN resource ``name`` field.
# Names verified May 2026 via package_show.
STATIONS_RESOURCE_HINT: str = "station reference"
HOURLY_RESOURCE_HINT: str = "hourly permanent"

# Plausible names for the date column in the hourly table.
DATE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "date",
    "Date",
    "DATE",
    "count_date",
    "trip_date",
    "TRAFFIC_COUNT_DATE",
    "traffic_count_date",
)


def _is_cache_fresh(out: Path, max_age_days: float) -> bool:
    if not out.exists():
        return False
    age_days = (time.time() - out.stat().st_mtime) / 86400.0
    return age_days < max_age_days


def _find_resource(resources: list[dict[str, Any]], hint: str) -> dict[str, Any]:
    """Match a CKAN resource by case-insensitive substring on its ``name``."""
    needle = hint.casefold()
    matches = [r for r in resources if needle in str(r.get("name", "")).casefold()]
    if not matches:
        names = [str(r.get("name", "")) for r in resources]
        raise RuntimeError(f"no resource matching {hint!r} (have: {names!r})")
    if len(matches) > 1:
        logger.warning(
            "multiple resources match %r; using first: %r",
            hint,
            [r.get("name") for r in matches],
        )
    return matches[0]


def _collect_datastore(api_root: str, resource_id: str, page_size: int = 10000) -> pd.DataFrame:
    batches: list[pd.DataFrame] = []
    for batch in _ckan.iter_datastore(api_root, resource_id, page_size=page_size):
        batches.append(pd.DataFrame.from_records(batch))
    if not batches:
        return pd.DataFrame()
    return pd.concat(batches, ignore_index=True)


def _read_zip_csvs(payload: bytes) -> pd.DataFrame:
    """Extract every CSV inside a ZIP and concatenate."""
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as fh:
                logger.info("reading %s from ZIP (%d bytes)", name, zf.getinfo(name).file_size)
                frames.append(pd.read_csv(fh, low_memory=False, dtype=str))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _filter_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter the hourly frame to ``[start, end]`` if a date column is present."""
    date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in df.columns), None)
    if date_col is None:
        logger.warning(
            "no recognised date column in hourly counts (columns=%s); skipping date filter",
            list(df.columns),
        )
        return df

    parsed = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    mask = parsed.between(pd.to_datetime(start), pd.to_datetime(end), inclusive="both")
    filtered: pd.DataFrame = df.loc[mask].copy()
    return filtered


def _fetch_resource_table(resource: dict[str, Any], api_root: str, page_size: int) -> pd.DataFrame:
    """Read a CKAN resource into a DataFrame, choosing strategy by metadata.

    - ``datastore_active=True`` → paginate via ``datastore_search``.
    - format ZIP → download the file and extract CSV(s) inside.
    - format CSV (no datastore) → download the file and read directly.
    """
    fmt = str(resource.get("format", "")).strip().casefold()
    rid = str(resource["id"])
    url = str(resource.get("url", ""))

    if resource.get("datastore_active"):
        logger.info("fetching %s as datastore (resource_id=%s)", resource.get("name"), rid)
        return _collect_datastore(api_root, rid, page_size=page_size)

    if fmt == "zip" or url.lower().endswith(".zip"):
        logger.info("downloading ZIP %s", url)
        payload = _ckan.download_bytes(url)
        return _read_zip_csvs(payload)

    if fmt in {"csv", "text/csv"} or url.lower().endswith(".csv"):
        logger.info("downloading CSV %s", url)
        payload = _ckan.download_bytes(url)
        return pd.read_csv(io.BytesIO(payload), low_memory=False, dtype=str)

    raise RuntimeError(
        f"don't know how to read resource {resource.get('name')!r} (format={fmt!r}, url={url!r})"
    )


def fetch(
    start: str,
    end: str,
    out_dir: Path,
    *,
    force: bool = False,
    max_age_days: float = 1.0,
    page_size: int = 10000,
) -> None:
    """Fetch the traffic reference + hourly tables to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stations_out = out_dir / "stations.parquet"
    hourly_out = out_dir / "hourly.parquet"

    if (
        not force
        and _is_cache_fresh(stations_out, max_age_days)
        and _is_cache_fresh(hourly_out, max_age_days)
    ):
        logger.info("cache hit for traffic fetch — skipping")
        return

    logger.info("looking up TfNSW package %s at %s", PACKAGE_ID, API_ROOT)
    package = _ckan.package_show(API_ROOT, PACKAGE_ID)
    resources = package.get("resources", [])
    if not resources:
        raise RuntimeError(f"package {PACKAGE_ID} has no resources")

    stations_resource = _find_resource(resources, STATIONS_RESOURCE_HINT)
    hourly_resource = _find_resource(resources, HOURLY_RESOURCE_HINT)

    stations_df = _fetch_resource_table(stations_resource, API_ROOT, page_size)
    stations_df.to_parquet(stations_out, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d station rows to %s", len(stations_df), stations_out)

    hourly_df = _fetch_resource_table(hourly_resource, API_ROOT, page_size)
    if not hourly_df.empty:
        hourly_df = _filter_by_date(hourly_df, start, end)
    hourly_df.to_parquet(hourly_out, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d hourly rows to %s", len(hourly_df), hourly_out)


def _find_resource_id(resources: list[dict[str, Any]], hint: str) -> str:
    """Backwards-compatible helper used by older tests."""
    return str(_find_resource(resources, hint)["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-age-days", type=float, default=1.0)
    parser.add_argument("--page-size", type=int, default=10000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    fetch(
        args.start,
        args.end,
        args.out,
        force=args.force,
        max_age_days=args.max_age_days,
        page_size=args.page_size,
    )


if __name__ == "__main__":
    main()
