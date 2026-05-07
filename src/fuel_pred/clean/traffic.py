"""Clean TfNSW traffic counts: hourly → daily, station reference normalised.

Reads:
    - ``data/raw/traffic/stations.parquet``  (Road Traffic Counts Station Reference)
    - ``data/raw/traffic/hourly.parquet``    (Road Traffic Counts Hourly Permanent)

Writes:
    - ``data/interim/traffic_daily.parquet``    (station_key, date, daily_total)
    - ``data/interim/traffic_stations.parquet`` (slim reference: station_key, lat, lon,
                                                 suburb, post_code, road_name, lga,
                                                 quality_rating, permanent_station)

Aggregation choices (verified against May-2026 data):

- The hourly resource is *already* one row per `(station_key, date,
  traffic_direction_seq, classification_seq)` — it is not literally
  hourly. Each row carries `daily_total` plus `hour_00`..`hour_23` columns.
- Up to 4 rows per `(station_key, date)` exist due to direction (0/1)
  by classification (observed values: 0, 2, 3).
- **Classification scheme is per-station**, not uniform across the
  dataset. In the Aug-2024 sample, ~210 stations emit only the
  vehicle-class breakdown (`classification_seq=2` light + `=3` heavy)
  while ~2 stations emit a pre-totalled row (`classification_seq=0`).
  No station mixes schemes. Summing `daily_total` across every row
  for a given `(station_key, date)` therefore yields the correct total
  in both cases — class 0 stations contribute one summand, breakdown
  stations contribute two summands that re-aggregate to the same total.
  TODO(spec): cross-check against the dataset's "Dataset Documentation"
  PDF if precise classification semantics matter for any downstream
  analysis.

Quality filter (per spec §12 Phase 2):

- Drop rows from non-permanent stations (`permanent_station != '1'`).
- Drop rows where `quality_rating < 3`. The TfNSW Data Quality
  Statement scales quality 1-5; ratings 1-2 indicate sparse coverage
  that produces unreliable daily totals.

Spec: spec.md §5.1.3, §7.4.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Quality rating cutoff. Values < 3 indicate sparse coverage per TfNSW's
# Data Quality Statement.
MIN_QUALITY_RATING: int = 3

STATION_KEEP_COLUMNS: tuple[str, ...] = (
    "station_key",
    "station_id",
    "name",
    "road_name",
    "suburb",
    "post_code",
    "lga",
    "rms_region",
    "wgs84_latitude",
    "wgs84_longitude",
    "quality_rating",
    "permanent_station",
    "road_functional_hierarchy",
)


def _slim_stations(stations: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns we'll need downstream + apply the quality filter."""
    keep = [c for c in STATION_KEEP_COLUMNS if c in stations.columns]
    out = stations[keep].copy()

    # Coerce numeric columns we filter on.
    out["quality_rating"] = pd.to_numeric(out["quality_rating"], errors="coerce")
    out["permanent_station"] = out["permanent_station"].astype(str).str.strip()
    out["wgs84_latitude"] = pd.to_numeric(out["wgs84_latitude"], errors="coerce")
    out["wgs84_longitude"] = pd.to_numeric(out["wgs84_longitude"], errors="coerce")

    before = len(out)
    out = out[
        (out["permanent_station"] == "1")
        & (out["quality_rating"] >= MIN_QUALITY_RATING)
        & out["wgs84_latitude"].notna()
        & out["wgs84_longitude"].notna()
    ].copy()
    logger.info(
        "stations: kept %d / %d (permanent + quality_rating >= %d + has lat/lon)",
        len(out),
        before,
        MIN_QUALITY_RATING,
    )
    return out


def _aggregate_daily(hourly: pd.DataFrame, kept_keys: pd.Series) -> pd.DataFrame:
    """Sum `daily_total` across direction + classification, keep good stations.

    See module docstring for the rationale: per-station classification
    schemes mean a blanket SUM across rows is the right roll-up.
    """
    required = {"station_key", "date", "daily_total"}
    missing = required - set(hourly.columns)
    if missing:
        raise RuntimeError(
            f"hourly missing required columns {missing}; schema may have changed"
        )

    hourly = hourly.copy()
    hourly["station_key"] = hourly["station_key"].astype(str).str.strip()
    hourly["daily_total"] = pd.to_numeric(hourly["daily_total"], errors="coerce")
    hourly["date"] = (
        pd.to_datetime(hourly["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.date
    )

    before = len(hourly)
    hourly = hourly[hourly["daily_total"].notna() & hourly["date"].notna()]
    logger.info("hourly: %d / %d rows after null filter", len(hourly), before)

    keep_set = set(kept_keys.astype(str))
    hourly = hourly[hourly["station_key"].isin(keep_set)]
    logger.info("hourly: %d rows after station-quality filter", len(hourly))

    daily: pd.DataFrame = (
        hourly.groupby(["station_key", "date"], as_index=False, observed=True)
        .agg(daily_total=("daily_total", "sum"))
    )
    return daily


def clean(in_dir: Path, out: Path, stations_out: Path) -> None:
    """Aggregate the TfNSW raw fetch to a daily-rolled clean view.

    Args:
        in_dir: directory with `stations.parquet` + `hourly.parquet`.
        out: where to write `traffic_daily.parquet`.
        stations_out: where to write `traffic_stations.parquet`.
    """
    stations_path = in_dir / "stations.parquet"
    hourly_path = in_dir / "hourly.parquet"
    if not stations_path.exists():
        raise RuntimeError(f"missing {stations_path}")
    if not hourly_path.exists():
        raise RuntimeError(f"missing {hourly_path}")

    stations_raw = pd.read_parquet(stations_path)
    hourly_raw = pd.read_parquet(hourly_path)

    stations = _slim_stations(stations_raw)
    daily = _aggregate_daily(hourly_raw, stations["station_key"].astype(str))

    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d daily rows to %s", len(daily), out)

    stations_out.parent.mkdir(parents=True, exist_ok=True)
    stations.to_parquet(stations_out, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d stations to %s", len(stations), stations_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stations-out", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    clean(args.in_dir, args.out, args.stations_out)


if __name__ == "__main__":
    main()
