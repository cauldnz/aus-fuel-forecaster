"""Per-station spatial join against the major-venues pilot list (spec §13.6).

Computes, for each station, its haversine distance and identity of the
nearest pilot venue plus a count of venues within ``radius_km``. The 10
venues are intentionally small (greater Sydney + Newcastle) so a pure-numpy
haversine over the cartesian product is trivially fast — no BallTree needed.

Output schema (``data/interim/stations_venues.parquet``):

    station_id, stn_nearest_venue_km, stn_nearest_venue_id,
    stn_nearest_venue_capacity, stn_nearest_venue_type,
    stn_n_venues_within_5km

Spec: spec.md §13.6; research doc
``docs/research/2026-05_major_events_features.md``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fuel_pred import config

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM: float = 6371.0
DEFAULT_RADIUS_KM: float = 5.0


def _haversine_km_matrix(
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    venue_lats: np.ndarray,
    venue_lons: np.ndarray,
) -> np.ndarray:
    """Pairwise haversine distance, returning an ``(n_stations, n_venues)`` km matrix.

    Vectorised numpy: builds the lat/lon broadcast pair and computes the
    great-circle distance in a single call. With ~5k stations × 10
    venues, the resulting 50k-cell matrix fits in <1 MB.
    """
    s_lat = np.radians(station_lats).reshape(-1, 1)
    s_lon = np.radians(station_lons).reshape(-1, 1)
    v_lat = np.radians(venue_lats).reshape(1, -1)
    v_lon = np.radians(venue_lons).reshape(1, -1)
    dlat = v_lat - s_lat
    dlon = v_lon - s_lon
    a = np.sin(dlat / 2) ** 2 + np.cos(s_lat) * np.cos(v_lat) * np.sin(dlon / 2) ** 2
    result: np.ndarray = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return result


def compute_station_venues(
    stations: pd.DataFrame,
    venues: pd.DataFrame,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> pd.DataFrame:
    """For each station, compute nearest-venue distance/id/capacity/type + radius count.

    Args:
        stations: must have ``station_id``, ``lat``, ``lon``. Rows with
            NaN lat/lon are kept in the output but get null values for
            every venue feature column.
        venues: from ``data/static/major_venues.csv`` — must have
            ``venue_id``, ``lat``, ``lon``, ``capacity``, ``type``.
        radius_km: radius for the per-station "venues within X km"
            count. Default 5 km per the pilot research doc.

    Returns:
        DataFrame with columns ``station_id, stn_nearest_venue_km,
        stn_nearest_venue_id, stn_nearest_venue_capacity,
        stn_nearest_venue_type, stn_n_venues_within_<radius>km``.
        One row per input station; stations without coords get null
        distance/id/capacity/type and 0 for the radius count.
    """
    required_station_cols = {"station_id", "lat", "lon"}
    missing_s = required_station_cols - set(stations.columns)
    if missing_s:
        raise ValueError(f"stations missing required columns: {sorted(missing_s)}")
    required_venue_cols = {"venue_id", "lat", "lon", "capacity", "type"}
    missing_v = required_venue_cols - set(venues.columns)
    if missing_v:
        raise ValueError(f"venues missing required columns: {sorted(missing_v)}")
    if venues.empty:
        raise ValueError("venues frame is empty — cannot compute nearest")

    radius_col = f"stn_n_venues_within_{int(radius_km)}km"

    # Keep stations with usable coords; rest get null/zero rows joined back at the end.
    s_mask = stations["lat"].notna() & stations["lon"].notna()
    s = stations.loc[s_mask, ["station_id", "lat", "lon"]].reset_index(drop=True)
    n_skipped = int((~s_mask).sum())
    if n_skipped:
        logger.warning(
            "%d / %d stations missing lat/lon — venue features will be null",
            n_skipped,
            len(stations),
        )

    if s.empty:
        logger.warning("no stations with usable lat/lon — emitting all-null result")
        return _empty_result(stations, radius_col)

    dist_matrix = _haversine_km_matrix(
        s["lat"].to_numpy(dtype=np.float64),
        s["lon"].to_numpy(dtype=np.float64),
        venues["lat"].to_numpy(dtype=np.float64),
        venues["lon"].to_numpy(dtype=np.float64),
    )
    nearest_idx = np.argmin(dist_matrix, axis=1)
    nearest_km = dist_matrix[np.arange(len(s)), nearest_idx]
    within = (dist_matrix <= radius_km).sum(axis=1).astype(np.int64)

    venue_ids = venues["venue_id"].to_numpy()
    venue_caps = venues["capacity"].to_numpy()
    venue_types = venues["type"].to_numpy()

    joined = pd.DataFrame(
        {
            "station_id": s["station_id"].to_numpy(),
            "stn_nearest_venue_km": nearest_km.astype(np.float64),
            "stn_nearest_venue_id": venue_ids[nearest_idx],
            "stn_nearest_venue_capacity": venue_caps[nearest_idx].astype(np.float64),
            "stn_nearest_venue_type": venue_types[nearest_idx],
            radius_col: within,
        }
    )

    # Re-attach any skipped stations as null/zero rows so the output
    # has one row per input station (matches spatial.nearest's convention).
    full = stations[["station_id"]].merge(joined, on="station_id", how="left")
    full[radius_col] = full[radius_col].fillna(0).astype(np.int64)

    logger.info(
        "computed venue features for %d / %d stations (%d skipped); "
        "nearest distance median = %.2f km, max = %.2f km",
        len(s),
        len(stations),
        n_skipped,
        float(np.median(nearest_km)),
        float(np.max(nearest_km)),
    )
    return full


def _empty_result(stations: pd.DataFrame, radius_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": stations["station_id"].to_numpy(),
            "stn_nearest_venue_km": np.full(len(stations), np.nan, dtype=np.float64),
            "stn_nearest_venue_id": pd.Series([pd.NA] * len(stations), dtype="object"),
            "stn_nearest_venue_capacity": np.full(len(stations), np.nan, dtype=np.float64),
            "stn_nearest_venue_type": pd.Series([pd.NA] * len(stations), dtype="object"),
            radius_col: np.zeros(len(stations), dtype=np.int64),
        }
    )


def compute_station_venues_from_paths(
    stations_path: Path,
    venues_path: Path,
    out_path: Path,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> None:
    """File-IO wrapper around :func:`compute_station_venues`."""
    logger.info("loading stations from %s", stations_path)
    stations = pd.read_parquet(stations_path)
    logger.info("loading venues from %s", venues_path)
    venues = pd.read_csv(venues_path)
    logger.info(
        "computing venue features: %d stations × %d venues (radius_km=%g)",
        len(stations),
        len(venues),
        radius_km,
    )

    result = compute_station_venues(stations, venues, radius_km=radius_km)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d rows × %d cols to %s", len(result), len(result.columns), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", required=True, type=Path)
    parser.add_argument(
        "--venues",
        type=Path,
        default=config.STATIC_MAJOR_VENUES,
        help=f"default: {config.STATIC_MAJOR_VENUES}",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--radius-km",
        type=float,
        default=DEFAULT_RADIUS_KM,
        help=f"radius for the 'venues within X km' count (default: {DEFAULT_RADIUS_KM})",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    compute_station_venues_from_paths(
        args.stations, args.venues, args.out, radius_km=args.radius_km
    )


if __name__ == "__main__":
    main()
