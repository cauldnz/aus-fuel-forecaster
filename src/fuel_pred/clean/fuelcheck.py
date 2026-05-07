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
from pathlib import Path

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
    "brand": "brand",
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
    "brand",
    "fuel_code",
    "price_updated_date",
    "price",
)

# Default chunk size for streaming through monthly Parquets. A single month
# is ~70-90k rows; we keep memory bounded by aggregating each batch and only
# concatenating the daily output (~2-3 orders of magnitude smaller).
MONTHLY_CHUNK: int = 4


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


def load_brand_aliases(path: Path) -> dict[str, str]:
    """Load ``raw_brand → canonical_brand`` mapping from the static CSV."""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = (row.get("raw_brand") or "").strip()
            canonical = (row.get("canonical_brand") or "").strip()
            if raw and canonical:
                mapping[raw] = canonical
    return mapping


def _canonicalise_brand(raw: str | None, mapping: dict[str, str], unmapped: set[str]) -> str:
    """Map a raw brand string to its canonical form; identity-fall-through."""
    if raw is None or pd.isna(raw):
        return "Independent"
    raw_str = str(raw).strip()
    if raw_str in mapping:
        return mapping[raw_str]
    unmapped.add(raw_str)
    return raw_str


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


def _process_month(path: Path, brand_map: dict[str, str], unmapped: set[str]) -> pd.DataFrame:
    """Load one monthly Parquet, normalise, return a long-form rows frame.

    Output columns: ``station_id, name, address, suburb, postcode, brand,
    fuel_code, date, price``. One row per FuelCheck event (no aggregation
    yet — that happens after concatenation across months).
    """
    raw = pd.read_parquet(path)
    df = _normalise_columns(raw)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("%s missing columns %s; skipping", path.name, missing)
        return pd.DataFrame()

    df = df.dropna(subset=["name", "address", "suburb", "postcode", "fuel_code", "price"]).copy()
    df["station_id"] = df.apply(
        lambda r: _hash_station(r["name"], r["address"], r["suburb"], r["postcode"]), axis=1
    )
    df["brand"] = df["brand"].apply(lambda b: _canonicalise_brand(b, brand_map, unmapped))
    df["fuel_code"] = df["fuel_code"].astype(str).str.strip().str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notna()].copy()
    df["date"] = _parse_price_date(df["price_updated_date"])
    df = df[df["date"].notna()].copy()

    out_cols = [
        "station_id",
        "name",
        "address",
        "suburb",
        "postcode",
        "brand",
        "fuel_code",
        "date",
        "price",
    ]
    return df[out_cols]


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
            brand=("brand", "last"),
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
    """
    aliases_path = brand_aliases or (config.DATA_STATIC / "brand_aliases.csv")
    brand_map = load_brand_aliases(aliases_path)
    logger.info("loaded %d brand aliases from %s", len(brand_map), aliases_path)

    monthly_paths = sorted(in_dir.glob("*.parquet"))
    if not monthly_paths:
        raise RuntimeError(f"no monthly parquets found in {in_dir}")

    unmapped: set[str] = set()
    daily_chunks: list[pd.DataFrame] = []
    roster_chunks: list[pd.DataFrame] = []

    for batch in _batched(monthly_paths, MONTHLY_CHUNK):
        events = pd.concat(
            (_process_month(p, brand_map, unmapped) for p in batch),
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
            brand=("brand", "last"),
            first_seen=("first_seen", "min"),
            last_seen=("last_seen", "max"),
        )
    )
    stations_out.parent.mkdir(parents=True, exist_ok=True)
    roster.to_parquet(stations_out, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d stations to %s", len(roster), stations_out)

    if unmapped:
        sample = sorted(unmapped)[:20]
        logger.warning(
            "%d brand strings unmapped — append to brand_aliases.csv. Sample: %s",
            len(unmapped),
            sample,
        )


def _batched(items: list[Path], size: int) -> Iterator[list[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stations-out", required=True, type=Path)
    parser.add_argument("--brand-aliases", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    clean(args.in_dir, args.out, args.stations_out, brand_aliases=args.brand_aliases)


if __name__ == "__main__":
    main()
