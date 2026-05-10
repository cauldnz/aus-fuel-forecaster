"""Compute nearest-neighbour assignments: each fuel station to its top-N traffic counters.

Output schema (`data/interim/station_to_counter.parquet`):

    station_id, counter_rank, counter_id, distance_km

Per spec §7.4 the feature builder uses the top-3 closest counters; we
compute top-5 here so the feature builder can pick any subset without
re-running the spatial join. Includes the radius-count column
(`stn_n_counters_within_5km`) which is also used by §7.4.

Plus a per-station `stn_distance_to_sydney_terminal_km` column written
to a separate summary parquet for §7.5. Botany terminal coords hardcoded
here — Australian fuel terminal locations don't change.

Spec: spec.md §7.4, §7.5.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)

# Sydney fuel terminal — Botany. Distance feature uses this anchor.
SYDNEY_TERMINAL_LAT: float = -33.9619
SYDNEY_TERMINAL_LON: float = 151.2095

# Default number of top-N counters to record per station. Spec §7.4 uses
# top-3; we keep 5 in the table so feature builders can experiment.
DEFAULT_TOP_N: int = 5

# Radius for the "counters within X km" count column (spec §7.4).
RADIUS_KM: float = 5.0

# Mean Earth radius (km). BallTree haversine returns radians; multiply.
EARTH_RADIUS_KM: float = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in km."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _radians_array(df: pd.DataFrame, lat_col: str, lon_col: str) -> np.ndarray:
    """Convert (lat, lon) columns to a (N, 2) float64 array of radians."""
    arr = df[[lat_col, lon_col]].to_numpy(dtype=np.float64)
    radians: np.ndarray = np.radians(arr)
    return radians


def compute_top_n(
    stations: pd.DataFrame,
    counters: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
    radius_km: float = RADIUS_KM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the top-N table + a per-station distance/count summary.

    Args:
        stations: must have columns `station_id`, `lat`, `lon`. Rows
            with NaN lat/lon are silently skipped.
        counters: TfNSW traffic-counter reference table from
            `clean.traffic`. Must have `station_key`, `wgs84_latitude`,
            `wgs84_longitude` (`station_key` is the FK used to join
            against `traffic_daily.parquet`).
        top_n: number of nearest counters to record per fuel station.
        radius_km: radius for the "counters within X km" count.

    Returns:
        `(top_table, summary)`:
        - `top_table` has one row per `(station_id, rank)` for ranks
          1..top_n: `station_id, counter_rank, counter_id, distance_km`.
          Stations with no usable lat/lon are omitted.
        - `summary` has one row per fuel station:
          `station_id, stn_distance_to_sydney_terminal_km,
          stn_n_counters_within_<radius>km`.
    """
    if "station_id" not in stations.columns:
        raise ValueError("stations must have a 'station_id' column")
    if "station_key" not in counters.columns:
        raise ValueError("counters must have a 'station_key' column (TfNSW FK)")

    # Drop rows the spatial join can't use; surface counts.
    s_mask = stations["lat"].notna() & stations["lon"].notna()
    c_mask = counters["wgs84_latitude"].notna() & counters["wgs84_longitude"].notna()
    s = stations.loc[s_mask, ["station_id", "lat", "lon"]].reset_index(drop=True)
    c = counters.loc[
        c_mask, ["station_key", "wgs84_latitude", "wgs84_longitude"]
    ].reset_index(drop=True)

    n_stations_skipped = int((~s_mask).sum())
    n_counters_skipped = int((~c_mask).sum())
    if n_stations_skipped:
        logger.warning("skipping %d fuel stations with missing lat/lon", n_stations_skipped)
    if n_counters_skipped:
        logger.warning("skipping %d traffic counters with missing lat/lon", n_counters_skipped)

    if len(c) == 0:
        raise RuntimeError("no traffic counters with usable lat/lon — cannot compute nearest")
    if len(s) == 0:
        logger.warning("no fuel stations with usable lat/lon — empty output")
        return _empty_top_table(), _empty_summary(stations, radius_km)

    # BallTree expects radians. Effective top_n bounded by counter pool size.
    counter_rad = _radians_array(c, "wgs84_latitude", "wgs84_longitude")
    station_rad = _radians_array(s, "lat", "lon")
    k = min(top_n, len(c))

    tree = BallTree(counter_rad, metric="haversine")
    distances_rad, indices = tree.query(station_rad, k=k)
    distances_km = distances_rad * EARTH_RADIUS_KM

    # Build the long-form top-N table.
    rows = []
    for s_pos, station_id in enumerate(s["station_id"]):
        for rank, (idx, dkm) in enumerate(
            zip(indices[s_pos], distances_km[s_pos], strict=True), start=1
        ):
            rows.append(
                {
                    "station_id": station_id,
                    "counter_rank": rank,
                    "counter_id": str(c["station_key"].iloc[idx]),
                    "distance_km": float(dkm),
                }
            )
    top_table = pd.DataFrame(
        rows, columns=["station_id", "counter_rank", "counter_id", "distance_km"]
    )

    # Radius count: per-station, how many counters within `radius_km`.
    radius_rad = radius_km / EARTH_RADIUS_KM
    radius_counts = tree.query_radius(station_rad, r=radius_rad, count_only=True)
    radius_col = f"stn_n_counters_within_{int(radius_km)}km"

    # Sydney terminal distance: scalar haversine per station.
    terminal_dist = np.array(
        [
            _haversine_km(lat, lon, SYDNEY_TERMINAL_LAT, SYDNEY_TERMINAL_LON)
            for lat, lon in zip(s["lat"], s["lon"], strict=True)
        ]
    )

    summary_partial = pd.DataFrame(
        {
            "station_id": s["station_id"].to_numpy(),
            "stn_distance_to_sydney_terminal_km": terminal_dist,
            radius_col: radius_counts.astype(np.int64),
        }
    )
    # Re-attach the dropped stations with null distances + zero radius count.
    summary = stations[["station_id"]].merge(summary_partial, on="station_id", how="left")
    summary[radius_col] = summary[radius_col].fillna(0).astype(np.int64)

    return top_table, summary


def _empty_top_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": pd.Series(dtype="object"),
            "counter_rank": pd.Series(dtype="int64"),
            "counter_id": pd.Series(dtype="object"),
            "distance_km": pd.Series(dtype="float64"),
        }
    )


def _empty_summary(stations: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    radius_col = f"stn_n_counters_within_{int(radius_km)}km"
    return pd.DataFrame(
        {
            "station_id": stations["station_id"].to_numpy(),
            "stn_distance_to_sydney_terminal_km": np.full(len(stations), np.nan),
            radius_col: np.zeros(len(stations), dtype=np.int64),
        }
    )


def compute_nearest(
    stations_path: Path,
    counters_path: Path,
    out_path: Path,
    *,
    summary_out: Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    radius_km: float = RADIUS_KM,
) -> None:
    """Read both inputs, compute top-N + summary, write parquet outputs.

    Args:
        stations_path: stations.parquet (post Phase 2/3 — needs lat/lon).
        counters_path: traffic_stations.parquet (clean.traffic output).
        out_path: top-N long-form parquet.
        summary_out: per-station scalar summary (terminal distance,
            radius count). Defaults to `<out_path stem>_summary.parquet`
            beside `out_path`.
        top_n: number of nearest counters per station.
        radius_km: radius for the per-station counter-count column.
    """
    stations = pd.read_parquet(stations_path)
    counters = pd.read_parquet(counters_path)
    logger.info(
        "loaded %d fuel stations + %d traffic counters", len(stations), len(counters)
    )

    top_table, summary = compute_top_n(
        stations, counters, top_n=top_n, radius_km=radius_km
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    top_table.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d top-N rows to %s", len(top_table), out_path)

    summary_out = summary_out or out_path.with_name(out_path.stem + "_summary.parquet")
    summary.to_parquet(summary_out, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d summary rows to %s", len(summary), summary_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", required=True, type=Path)
    parser.add_argument("--counters", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--radius-km", type=float, default=RADIUS_KM)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    compute_nearest(
        args.stations,
        args.counters,
        args.out,
        summary_out=args.summary_out,
        top_n=args.top_n,
        radius_km=args.radius_km,
    )


if __name__ == "__main__":
    main()
