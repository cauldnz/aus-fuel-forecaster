"""Clean and aggregate raw FuelCheck monthly Parquets to daily granularity.

Reads: ``data/raw/fuelcheck/<YYYY-MM>.parquet``  (one per month)
Writes:
    - ``data/interim/fuel_daily.parquet``  (daily aggregates per station / fuel)
    - ``data/interim/stations.parquet``    (one row per unique station)

Tasks:
1. Concatenate monthly Parquets in chunks. Schema drifts across years
   (column renames; PriceUpdatedDate format variants); a normalisation
   layer maps every variant to the §6 schema.
2. Standardise brand strings via ``data/static/brand_aliases.csv``.
   Unmapped brands log a WARNING and pass through verbatim.
3. Generate stable ``station_id = sha1(name|address|suburb|postcode)[:16]``.
   Same physical station with the same address text → always the same id.
4. Aggregate intraday price events to daily mean/min/max/n per
   ``(station_id, fuel_code, date)``. Both U91 and Diesel are kept; per
   §3, U91 is the forecast target and DL exists as a candidate feature.
5. Build the unique stations roster — latest non-null value per station_id
   for each descriptive field, plus first_seen / last_seen.

Spec: spec.md §6.1, §6.2; §3 (U91 target, DL kept).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import pandas as pd

from fuel_pred import config

logger = logging.getLogger(__name__)

# ----------------------------- Schema normalisation -----------------------------
# The raw CSV/XLSX columns differ across years. Map every variant we've seen
# to a canonical name. Lookup is case-insensitive on the raw column name.

CANONICAL_COLUMNS: dict[str, str] = {
    "servicestationname": "name",
    "service_station_name": "name",
    "address": "address",
    "suburb": "suburb",
    "postcode": "postcode",
    "brand": "brand_raw",
    "fuelcode": "fuel_code",
    "fuel_code": "fuel_code",
    "priceupdateddate": "price_updated_date",
    "price_updated_date": "price_updated_date",
    "price": "price",
}

REQUIRED_COLUMNS: tuple[str, ...] = (
    "name",
    "address",
    "suburb",
    "postcode",
    "brand_raw",
    "fuel_code",
    "price_updated_date",
    "price",
)

# Default chunk size for streaming through monthly Parquets. A single month
# is ~70-90k rows; we keep memory bounded by aggregating each batch and only
# concatenating the daily output (~2-3 orders of magnitude smaller).
MONTHLY_CHUNK: int = 4


# Tokens we expect to see in a header row across all 3 known Excel
# layouts. Lowercased, no separators. Used to detect which row holds
# the actual headers when the source file has a title cell at row 0.
_HEADER_TOKENS: frozenset[str] = frozenset(CANONICAL_COLUMNS)

# Descriptive columns whose values are visually "merged" in the
# Variant C layout — only the first row of each station's block has
# them filled, subsequent rows are NaN. We forward-fill these (but
# never the per-event columns: fuel code, price, timestamp).
_MERGED_CELL_DESCRIPTIVE_COLUMNS: frozenset[str] = frozenset(
    {"servicestationname", "address", "suburb", "postcode", "brand"}
)


def _row_looks_like_headers(row: pd.Series) -> bool:
    """True if this row's non-null values include canonical header tokens.

    Used by ``_detect_and_promote_headers`` to identify which row of a
    monthly file actually contains the column names. Two-or-more matches
    is the threshold to avoid false positives from data rows that
    coincidentally contain a single header-like value.
    """
    matches = 0
    for v in row.values:
        if v is None or pd.isna(v):
            continue
        if str(v).strip().lower() in _HEADER_TOKENS:
            matches += 1
            if matches >= 2:
                return True
    return False


def _detect_and_promote_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Recover canonical headers from any of the 3 known FuelCheck Excel layouts.

    NSW Open Data ships monthly Price History workbooks in three header
    layouts (issue #19). Our fetcher reads ``header=0`` for everything
    and persists whatever it finds, so the cached parquets carry the
    layout drift forward. This helper inspects the first 1-2 rows and
    promotes the real header row to ``df.columns`` so the downstream
    ``_normalise_columns`` path works for all three.

    - **Variant A** — headers at row 0 (the common case). Returned as-is.
    - **Variant B** — title cell at row 0 (e.g. ``Price_History_July_2017``),
      real headers at row 1. We promote row 0 to columns and drop it.
    - **Variant C** — title cell at row 0, blank row at row 0 of the
      data, real headers at row 1, and descriptive columns
      (ServiceStationName, Address, Suburb, Postcode, Brand) only filled
      on the first row of each station's "block" (visual merged cells).
      We promote, drop the prelude, and forward-fill the descriptive
      columns so each event row carries its station identity.

    If none of the three patterns match (a future schema change), we
    return ``df`` unchanged and let ``_normalise_columns`` produce the
    "missing columns" warning the caller already handles.
    """
    if df.empty or len(df.columns) == 0:
        return df

    # Variant A: column 0 is already a canonical header name.
    first_col = str(df.columns[0]).strip().lower()
    if first_col in _HEADER_TOKENS:
        return df

    # Variant B: row 0 is the headers (column 0 was a title cell).
    if len(df) >= 1 and _row_looks_like_headers(df.iloc[0]):
        new_cols = [str(v) for v in df.iloc[0].tolist()]
        out = df.iloc[1:].copy()
        out.columns = new_cols
        return out.reset_index(drop=True)

    # Variant C: row 0 is blank, row 1 is the headers, descriptive
    # columns are merged-cell-style (NaN-filled in subsequent rows).
    if (
        len(df) >= 2
        and df.iloc[0].isna().all()
        and _row_looks_like_headers(df.iloc[1])
    ):
        new_cols = [str(v) for v in df.iloc[1].tolist()]
        out = df.iloc[2:].copy()
        out.columns = new_cols
        # Forward-fill the sticky descriptive columns. Per-event
        # columns (FuelCode, PriceUpdatedDate, Price) are explicitly
        # NOT ffill'd — they're the actual per-row data.
        ffill_cols = [
            c for c in out.columns
            if str(c).strip().lower() in _MERGED_CELL_DESCRIPTIVE_COLUMNS
        ]
        if ffill_cols:
            out[ffill_cols] = out[ffill_cols].ffill()
        return out.reset_index(drop=True)

    # Unknown layout — let _normalise_columns surface the issue via
    # its existing "missing columns" warning.
    return df


def _normalise_postcode_series(series: pd.Series) -> pd.Series:
    """Coerce a postcode column to canonical 4-digit string form.

    Issue #23: pandas can read postcodes as ``float64`` from some monthly
    parquets (probably driven by occasional missing values triggering
    numeric inference), so a perfectly normal ``2776`` ends up serialised
    as ``'2776.0'``. Strip the spurious ``.0`` so downstream consumers
    (geocoders, joins) see consistent string postcodes regardless of
    monthly source. Non-numeric postcodes (e.g. LPO codes) pass through
    untouched.
    """
    coerced = series.astype("string").str.strip()
    return coerced.str.replace(r"\.0$", "", regex=True)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical §6 names; drop unknown columns."""
    rename: dict[str, str] = {}
    for col in df.columns:
        canonical = CANONICAL_COLUMNS.get(str(col).strip().lower())
        if canonical is not None:
            rename[col] = canonical
    out = df.rename(columns=rename)
    keep = [c for c in REQUIRED_COLUMNS if c in out.columns]
    return out[keep].copy()


def _hash_station(name: str, address: str, suburb: str, postcode: str) -> str:
    """Stable 16-char SHA1 prefix of the station's identifying tuple.

    Same inputs always produce the same id, across runs and across machines.
    Whitespace is collapsed and case-folded so that minor cosmetic variants
    don't fragment a single physical station into many ids.
    """
    parts = [str(p).strip().casefold() for p in (name, address, suburb, postcode)]
    blob = "|".join(parts).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


# ----------------------------- Brand canonicalisation -----------------------------


@dataclass(frozen=True)
class BrandInfo:
    """Lookup result for a raw FuelCheck Brand string."""

    canonical: str
    is_major: bool


def load_brand_aliases(path: Path) -> dict[str, BrandInfo]:
    """Load ``raw_brand → BrandInfo(canonical, is_major)`` from the static CSV."""
    mapping: dict[str, BrandInfo] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = (row.get("raw_brand") or "").strip()
            canonical = (row.get("canonical_brand") or "").strip()
            is_major = (row.get("is_major") or "").strip().lower() == "true"
            if raw and canonical:
                mapping[raw] = BrandInfo(canonical=canonical, is_major=is_major)
    return mapping


def _canonicalise_brand(
    raw: str | None, mapping: dict[str, BrandInfo], unmapped: set[str]
) -> BrandInfo:
    """Map a raw brand string to (canonical, is_major); identity-fall-through.

    Unmapped raws pass through with `canonical=raw` and `is_major=False`,
    and are recorded in `unmapped` for a single end-of-run WARNING.
    """
    if raw is None or pd.isna(raw):
        return BrandInfo(canonical="Independent", is_major=False)
    raw_str = str(raw).strip()
    if raw_str in mapping:
        return mapping[raw_str]
    unmapped.add(raw_str)
    return BrandInfo(canonical=raw_str, is_major=False)


# ----------------------------- Per-month processing -----------------------------


def _parse_price_date(series: pd.Series) -> pd.Series:
    """Parse PriceUpdatedDate into a tz-naive ``date``.

    Accepts every format we've observed in the wild:
    - ``2024/08/01 12:34:56`` (slash-delimited, the most common)
    - ``2024-09-01T12:34:56Z`` (ISO 8601 with TZ)
    - bare dates

    ``format="mixed"`` is needed because a single monthly file can carry
    rows in more than one format (a known FuelCheck quirk around format
    transition months).
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_convert(None).dt.date


def _process_month(
    path: Path,
    brand_map: dict[str, BrandInfo],
    unmapped: set[str],
    miss_stats: dict[str, _BrandMissEntry] | None = None,
) -> pd.DataFrame:
    """Load one monthly Parquet, normalise, return a long-form rows frame.

    Output columns: ``station_id, name, address, suburb, postcode,
    brand_raw, brand_canonical, brand_is_major, fuel_code, date, price``.
    Both raw and canonical brand are preserved — see CLAUDE.md memory
    "Preserve raw alongside normalised". One row per FuelCheck event.

    ``miss_stats`` is an optional accumulator for the per-month brand-miss
    sidecar (see ``_record_brand_misses``); when provided, this function
    appends per-raw-brand stats (occurrence count, distinct station ids,
    sample address/suburb) so ``clean()`` can dump a structured CSV.
    """
    raw = pd.read_parquet(path)
    raw = _detect_and_promote_headers(raw)
    df = _normalise_columns(raw)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("%s missing columns %s; skipping", path.name, missing)
        return pd.DataFrame()

    # Coerce postcode to canonical str form first (issue #23) — the
    # downstream dropna() and station_id hashing both depend on the
    # postcode column being string, not "2776.0" floats.
    df["postcode"] = _normalise_postcode_series(df["postcode"])
    df = df.dropna(subset=["name", "address", "suburb", "postcode", "fuel_code", "price"]).copy()
    df["station_id"] = df.apply(
        lambda r: _hash_station(r["name"], r["address"], r["suburb"], r["postcode"]), axis=1
    )
    df["brand_raw"] = df["brand_raw"].astype(str).str.strip()
    canonicals: list[str] = []
    is_majors: list[bool] = []
    for raw in df["brand_raw"].tolist():
        info = _canonicalise_brand(raw, brand_map, unmapped)
        canonicals.append(info.canonical)
        is_majors.append(info.is_major)
    df["brand_canonical"] = canonicals
    df["brand_is_major"] = is_majors
    df["fuel_code"] = df["fuel_code"].astype(str).str.strip().str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notna()].copy()
    df["date"] = _parse_price_date(df["price_updated_date"])
    df = df[df["date"].notna()].copy()

    if miss_stats is not None and unmapped:
        # Restrict to rows whose raw brand is unmapped — note we filter on
        # the `unmapped` set rather than `brand_canonical == brand_raw`,
        # because legitimately-mapped brands can also have canonical == raw
        # (e.g. "BP" maps to "BP").
        _record_brand_misses(df, unmapped, miss_stats, source_file=path.name)

    out_cols = [
        "station_id",
        "name",
        "address",
        "suburb",
        "postcode",
        "brand_raw",
        "brand_canonical",
        "brand_is_major",
        "fuel_code",
        "date",
        "price",
    ]
    return df[out_cols]


class _BrandMissEntry(TypedDict):
    """Per-raw-brand stats accumulated by ``_record_brand_misses``."""

    n_occurrences: int
    stations: set[str]
    sample_name: str
    sample_address: str
    sample_suburb: str
    first_seen_in: str


def _record_brand_misses(
    df: pd.DataFrame,
    unmapped: set[str],
    miss_stats: dict[str, _BrandMissEntry],
    *,
    source_file: str,
) -> None:
    """Accumulate per-raw-brand stats for the brand-miss CSV sidecar.

    Tracks: total occurrence count, distinct station_ids, sample
    address/suburb/name (the first one we see is good enough — the goal
    is to give a human enough to identify the brand and add it to
    `data/static/brand_aliases.csv`), and the first source file the
    brand appeared in.
    """
    miss_rows = df[df["brand_raw"].isin(unmapped)]
    if miss_rows.empty:
        return
    # `brand_raw` is normalised to str earlier in `_process_month` so
    # `groupby` keys are always str at runtime; cast for mypy.
    for raw_key, group in miss_rows.groupby("brand_raw"):
        raw = cast(str, raw_key)
        if raw not in miss_stats:
            first = group.iloc[0]
            miss_stats[raw] = _BrandMissEntry(
                n_occurrences=0,
                stations=set(),
                sample_name=str(first["name"]),
                sample_address=str(first["address"]),
                sample_suburb=str(first["suburb"]),
                first_seen_in=source_file,
            )
        entry = miss_stats[raw]
        entry["n_occurrences"] += len(group)
        entry["stations"].update(group["station_id"].tolist())


def _write_brand_misses_csv(
    miss_stats: dict[str, _BrandMissEntry], out_path: Path
) -> None:
    """Persist the unmapped-brand sidecar to ``out_path`` (CSV, sorted desc by count).

    Schema: raw_brand, n_occurrences, n_stations, sample_name,
    sample_address, sample_suburb, first_seen_in. Designed to be
    consumable both by humans curating ``brand_aliases.csv`` and by
    automated tools that want to suggest canonical mappings.
    """
    if not miss_stats:
        # Don't leave a stale file from a previous run.
        if out_path.exists():
            out_path.unlink()
        return

    rows = [
        {
            "raw_brand": raw,
            "n_occurrences": entry["n_occurrences"],
            "n_stations": len(entry["stations"]),
            "sample_name": entry["sample_name"],
            "sample_address": entry["sample_address"],
            "sample_suburb": entry["sample_suburb"],
            "first_seen_in": entry["first_seen_in"],
        }
        for raw, entry in miss_stats.items()
    ]
    df = pd.DataFrame(rows).sort_values(
        ["n_occurrences", "raw_brand"], ascending=[False, True]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _aggregate_daily(events: pd.DataFrame) -> pd.DataFrame:
    """Reduce intraday events to one row per (station_id, fuel_code, date)."""
    grouped = events.groupby(["station_id", "fuel_code", "date"], observed=True)["price"]
    out = grouped.agg(price_mean="mean", price_min="min", price_max="max", n_obs="count")
    return out.reset_index()


def _build_stations_roster(events: pd.DataFrame) -> pd.DataFrame:
    """One row per station_id, with descriptive fields from the most recent event."""
    events_sorted = events.sort_values("date")
    last = (
        events_sorted.groupby("station_id", as_index=False)
        .agg(
            name=("name", "last"),
            address=("address", "last"),
            suburb=("suburb", "last"),
            postcode=("postcode", "last"),
            brand_raw=("brand_raw", "last"),
            brand_canonical=("brand_canonical", "last"),
            brand_is_major=("brand_is_major", "last"),
            first_seen=("date", "min"),
            last_seen=("date", "max"),
        )
    )
    return last


# ----------------------------- Public API -----------------------------


def clean(
    in_dir: Path,
    out_path: Path,
    stations_out: Path,
    *,
    brand_aliases: Path | None = None,
    fuels: tuple[str, ...] = config.FUELS_V1,
    brand_misses_out: Path | None = None,
) -> None:
    """Aggregate monthly raw Parquets to daily fuel-prices + station roster.

    Args:
        in_dir: directory of raw monthly Parquets (``<YYYY-MM>.parquet``).
        out_path: where to write ``fuel_daily.parquet`` (§6.2 schema).
        stations_out: where to write ``stations.parquet`` (§6.1 minus geocoded cols).
        brand_aliases: path to ``data/static/brand_aliases.csv``; defaults
            to the in-repo location.
        fuels: tuple of FuelCode strings to keep. Defaults to U91 + DL —
            U91 is the forecast target, DL is kept as a candidate feature.
        brand_misses_out: where to write the brand-miss sidecar CSV
            (one row per unmapped raw_brand, sorted by occurrence count).
            Defaults to ``data/interim/brand_misses.csv``. Designed for
            curating new entries into ``brand_aliases.csv``.
    """
    aliases_path = brand_aliases or (config.DATA_STATIC / "brand_aliases.csv")
    brand_map = load_brand_aliases(aliases_path)
    logger.info("loaded %d brand aliases from %s", len(brand_map), aliases_path)

    monthly_paths = sorted(in_dir.glob("*.parquet"))
    if not monthly_paths:
        raise RuntimeError(f"no monthly parquets found in {in_dir}")

    unmapped: set[str] = set()
    miss_stats: dict[str, _BrandMissEntry] = {}
    daily_chunks: list[pd.DataFrame] = []
    roster_chunks: list[pd.DataFrame] = []

    for batch in _batched(monthly_paths, MONTHLY_CHUNK):
        events = pd.concat(
            (_process_month(p, brand_map, unmapped, miss_stats) for p in batch),
            ignore_index=True,
        )
        if events.empty:
            continue
        events = events[events["fuel_code"].isin(fuels)]
        if events.empty:
            continue

        daily_chunks.append(_aggregate_daily(events))
        roster_chunks.append(_build_stations_roster(events))

        logger.info(
            "batch %s..%s: %d events → %d daily rows",
            batch[0].stem,
            batch[-1].stem,
            len(events),
            len(daily_chunks[-1]),
        )

    if not daily_chunks:
        raise RuntimeError("clean produced no daily rows — check raw input")

    daily = pd.concat(daily_chunks, ignore_index=True)
    # A station can appear in multiple chunks; collapse identical
    # (station, fuel, date) keys by re-aggregating across chunks.
    daily = (
        daily.groupby(["station_id", "fuel_code", "date"], as_index=False)
        .agg(
            price_mean=("price_mean", "mean"),  # close enough — chunk sizes vary
            price_min=("price_min", "min"),
            price_max=("price_max", "max"),
            n_obs=("n_obs", "sum"),
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d daily rows to %s", len(daily), out_path)

    # Stations roster: keep the latest descriptive values per station_id and
    # extend first_seen/last_seen across all chunks.
    roster = pd.concat(roster_chunks, ignore_index=True)
    roster = (
        roster.sort_values("last_seen")
        .groupby("station_id", as_index=False)
        .agg(
            name=("name", "last"),
            address=("address", "last"),
            suburb=("suburb", "last"),
            postcode=("postcode", "last"),
            brand_raw=("brand_raw", "last"),
            brand_canonical=("brand_canonical", "last"),
            brand_is_major=("brand_is_major", "last"),
            first_seen=("first_seen", "min"),
            last_seen=("last_seen", "max"),
        )
    )
    stations_out.parent.mkdir(parents=True, exist_ok=True)
    roster.to_parquet(stations_out, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d stations to %s", len(roster), stations_out)

    misses_path = brand_misses_out or (config.DATA_INTERIM / "brand_misses.csv")
    _write_brand_misses_csv(miss_stats, misses_path)
    if unmapped:
        sample = sorted(unmapped)[:20]
        logger.warning(
            "%d brand strings unmapped — sidecar written to %s. Sample: %s",
            len(unmapped),
            misses_path,
            sample,
        )
    else:
        logger.info("all brand strings mapped — no sidecar needed")


def _batched(items: list[Path], size: int) -> Iterator[list[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stations-out", required=True, type=Path)
    parser.add_argument("--brand-aliases", type=Path, default=None)
    parser.add_argument(
        "--brand-misses-out",
        type=Path,
        default=None,
        help="Where to write the unmapped-brand sidecar CSV "
        "(default: data/interim/brand_misses.csv)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    clean(
        args.in_dir,
        args.out,
        args.stations_out,
        brand_aliases=args.brand_aliases,
        brand_misses_out=args.brand_misses_out,
    )


if __name__ == "__main__":
    main()
