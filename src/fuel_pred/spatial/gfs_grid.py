"""Compute per-station nearest-grid-point indices + bilinear weights for GFS/GEFS resolutions.

Output schema (`data/interim/station_grid_mapping.parquet`):

    station_id          str
    lat                 float    # original
    lon                 float    # original
    # GFS 0.25° (post-2021-04-01 window)
    gfs_lat_idx         int      # nearest grid point — lat index in the global GFS 0.25° grid
    gfs_lon_idx         int      # nearest grid point — lon index
    gfs_bl_lat_idx_0,1  int      # 2 surrounding lat indices (lower, upper)
    gfs_bl_lon_idx_0,1  int      # 2 surrounding lon indices (left, right)
    gfs_bl_w_00,01,10,11  float  # bilinear weights summing to 1.0
                                 # (00 = (lat_0, lon_0); 11 = (lat_1, lon_1))
    # GEFS 0.5° (2020-09-23 → 2021-03-31 bridge)
    gefs05_*                     # same six-tuple
    # GEFS 1° (2017-01-01 → 2020-09-22 backbone)
    gefs1_*                      # same six-tuple

NSW bounding box (lat in [-37.5, -28], lon in [140.5, 154]) is fully covered
by all three global grids. The mapping is computed once at pipeline init and
re-used by `fetch.gfs` (to size the per-(date, lead) grid parquet) and by
`build.make_features` (to bilinear-interpolate at panel-row stations).

The grid index conventions match how GFS/GEFS GRIB files lay out their
arrays after cfgrib parse:
    - latitudes are stored 90.0, 89.75, ..., -89.75, -90.0  (descending,
      step = `resolution`). `lat_idx = round((90 - lat) / res)`.
    - longitudes are stored 0.0, 0.25, ..., 359.75  (ascending east).
      Australia is at lons in [140, 154] — no wrap-around needed.
      `lon_idx = round(lon / res)`.

The bilinear-interp neighbours bracket the station. For a station at
lat ∈ (g_lat[i], g_lat[i+1]) (remembering lats decrease with i) and
lon ∈ (g_lon[j], g_lon[j+1]):

    w_00 = (1 - alpha_lat) * (1 - alpha_lon)
    w_01 = (1 - alpha_lat) *     alpha_lon
    w_10 =     alpha_lat   * (1 - alpha_lon)
    w_11 =     alpha_lat   *     alpha_lon

where `alpha_lat = (g_lat[i] - station_lat) / res` (since lats descend)
and `alpha_lon = (station_lon - g_lon[j]) / res`. All weights in [0, 1].

Sum of weights is exactly 1.0 (mathematically; rounding error <1e-15).

CLI:
    python -m fuel_pred.spatial.gfs_grid \\
        --stations data/interim/stations.parquet \\
        --out data/interim/station_grid_mapping.parquet

Spec: spec.md §13.7 (grid-cell caching), §13.8.
Research: docs/research/2026-06_nwp_archive_alternative.md §"Grid-cell caching architecture".
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Global lat extent in degrees (cfgrib reads GFS/GEFS with lats descending
# from 90 to -90). All three resolutions share these endpoints.
LAT_MAX: Final[float] = 90.0
LAT_MIN: Final[float] = -90.0
LON_MIN: Final[float] = 0.0
LON_MAX_EXCLUSIVE: Final[float] = 360.0  # grid wraps at 360 (= 0)

# Three native resolutions of the NOAA forecast products in scope (degrees).
# Keys match the per-resolution prefix used in the output schema:
#   gfs    -> 0.25°  GFS deterministic
#   gefs05 -> 0.5°   GEFS v12 ensemble mean (pgrb2ap5)
#   gefs1  -> 1.0°   GEFS pre-v12 control member (pgrb2a)
DEFAULT_RESOLUTIONS: Final[dict[str, float]] = {
    "gfs": 0.25,
    "gefs05": 0.5,
    "gefs1": 1.0,
}


def _lat_lon_axes(resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the global (lats_desc, lons_asc) coordinate axes for `resolution`.

    Lats descend from 90 to -90 inclusive (matches cfgrib's read order).
    Lons ascend from 0 to (360 - resolution) inclusive (no wrap).
    """
    # 90 .. -90 inclusive — both endpoints present, so n = 180/res + 1.
    n_lat = round((LAT_MAX - LAT_MIN) / resolution) + 1
    lats = np.linspace(LAT_MAX, LAT_MIN, n_lat, dtype=np.float64)

    # 0 .. (360 - res) inclusive — wraps at 360, so 360/res grid points.
    n_lon = round((LON_MAX_EXCLUSIVE - LON_MIN) / resolution)
    lons = np.linspace(LON_MIN, LON_MAX_EXCLUSIVE - resolution, n_lon, dtype=np.float64)

    return lats, lons


def _bilinear_for_station(
    lat: float, lon: float, resolution: float
) -> tuple[int, int, int, int, int, int, float, float, float, float]:
    """Compute 4-neighbour bilinear indices + weights for one station.

    Returns `(lat_idx_nearest, lon_idx_nearest, lat_idx_0, lat_idx_1,
    lon_idx_0, lon_idx_1, w_00, w_01, w_10, w_11)`.

    Indices are clamped to the global grid; weight sums to 1.0.

    Raises ValueError if lat/lon are outside [-90, 90] / [-180, 360).
    """
    if not (LAT_MIN <= lat <= LAT_MAX):
        raise ValueError(f"lat {lat} outside [-90, 90]")
    # Allow negative lons (some sources serve as -180..180); normalise.
    norm_lon = lon if lon >= 0 else lon + 360.0
    if not (LON_MIN <= norm_lon < LON_MAX_EXCLUSIVE):
        raise ValueError(f"lon {lon} (normalised {norm_lon}) outside [0, 360)")

    # Latitude index: lats descend. Index 0 = 90.0, step = -resolution.
    # Real-valued index: (90 - lat) / res
    f_lat = (LAT_MAX - lat) / resolution
    lat_idx_0 = int(np.floor(f_lat))
    lat_idx_1 = lat_idx_0 + 1
    alpha_lat = f_lat - lat_idx_0  # in [0, 1)

    # Clamp at the southern pole (lat = -90 exactly).
    n_lat = round((LAT_MAX - LAT_MIN) / resolution) + 1
    if lat_idx_1 >= n_lat:
        lat_idx_1 = n_lat - 1
        lat_idx_0 = n_lat - 1
        alpha_lat = 0.0

    # Longitude index: ascending from 0. Real-valued: lon / res.
    f_lon = norm_lon / resolution
    lon_idx_0 = int(np.floor(f_lon))
    lon_idx_1 = lon_idx_0 + 1
    alpha_lon = f_lon - lon_idx_0

    # Lon wraps at 360 (next grid point after 359.75 is 0.0 at 0.25°).
    n_lon = round((LON_MAX_EXCLUSIVE - LON_MIN) / resolution)
    if lon_idx_1 >= n_lon:
        lon_idx_1 = 0  # wrap east

    # Bilinear weights.
    w_00 = (1.0 - alpha_lat) * (1.0 - alpha_lon)
    w_01 = (1.0 - alpha_lat) * alpha_lon
    w_10 = alpha_lat * (1.0 - alpha_lon)
    w_11 = alpha_lat * alpha_lon

    # Nearest = the neighbour with the highest weight.
    weights = [(w_00, lat_idx_0, lon_idx_0),
               (w_01, lat_idx_0, lon_idx_1),
               (w_10, lat_idx_1, lon_idx_0),
               (w_11, lat_idx_1, lon_idx_1)]
    weights.sort(key=lambda t: -t[0])
    _, nearest_lat, nearest_lon = weights[0]

    return (
        nearest_lat, nearest_lon,
        lat_idx_0, lat_idx_1, lon_idx_0, lon_idx_1,
        w_00, w_01, w_10, w_11,
    )


def compute_station_grid_mapping(
    stations: pd.DataFrame,
    resolutions: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute per-station grid indices + bilinear weights for each resolution.

    Args:
        stations: DataFrame with columns `station_id`, `lat`, `lon`.
            Rows with NaN lat/lon are dropped with a warning.
        resolutions: prefix -> resolution-degrees mapping. Defaults to
            the 3 NOAA resolutions in scope (`gfs`/0.25, `gefs05`/0.5,
            `gefs1`/1.0). Each prefix becomes the column-name prefix
            in the output.

    Returns:
        DataFrame with columns `station_id, lat, lon` plus, for each
        resolution prefix `P`:
            `P_lat_idx, P_lon_idx` (nearest neighbour)
            `P_bl_lat_idx_0, P_bl_lat_idx_1` (2 surrounding lat indices)
            `P_bl_lon_idx_0, P_bl_lon_idx_1` (2 surrounding lon indices)
            `P_bl_w_00, P_bl_w_01, P_bl_w_10, P_bl_w_11` (weights, sum=1)

    The 4-neighbour scheme is enough for bilinear interp; nearest-neighbour
    is available as a degenerate case (the highest-weight neighbour).
    """
    if resolutions is None:
        resolutions = DEFAULT_RESOLUTIONS

    required = {"station_id", "lat", "lon"}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"stations missing required columns: {missing}")

    usable_mask = stations["lat"].notna() & stations["lon"].notna()
    n_skipped = int((~usable_mask).sum())
    if n_skipped:
        logger.warning("skipping %d stations with missing lat/lon", n_skipped)
    s = stations.loc[usable_mask, ["station_id", "lat", "lon"]].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for tup in s.itertuples(index=False):
        # itertuples emits a namedtuple; the lat/lon attrs are guaranteed
        # non-null by the mask above.
        lat = float(tup.lat)  # type: ignore[arg-type]
        lon = float(tup.lon)  # type: ignore[arg-type]
        row: dict[str, object] = {
            "station_id": str(tup.station_id),
            "lat": lat,
            "lon": lon,
        }
        for prefix, res in resolutions.items():
            (
                nearest_lat, nearest_lon,
                lat0, lat1, lon0, lon1,
                w00, w01, w10, w11,
            ) = _bilinear_for_station(lat, lon, res)
            row[f"{prefix}_lat_idx"] = nearest_lat
            row[f"{prefix}_lon_idx"] = nearest_lon
            row[f"{prefix}_bl_lat_idx_0"] = lat0
            row[f"{prefix}_bl_lat_idx_1"] = lat1
            row[f"{prefix}_bl_lon_idx_0"] = lon0
            row[f"{prefix}_bl_lon_idx_1"] = lon1
            row[f"{prefix}_bl_w_00"] = w00
            row[f"{prefix}_bl_w_01"] = w01
            row[f"{prefix}_bl_w_10"] = w10
            row[f"{prefix}_bl_w_11"] = w11
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "computed grid mapping for %d stations across %d resolutions: %s",
        len(df), len(resolutions), ", ".join(f"{p}={r}°" for p, r in resolutions.items()),
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", required=True, type=Path,
                        help="Parquet with station_id, lat, lon")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output parquet for the mapping table")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    stations = pd.read_parquet(args.stations, columns=["station_id", "lat", "lon"])
    logger.info("loaded %d stations from %s", len(stations), args.stations)

    mapping = compute_station_grid_mapping(stations)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: tmp + rename.
    tmp_path = args.out.with_suffix(args.out.suffix + ".tmp")
    mapping.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
    tmp_path.replace(args.out)
    logger.info("wrote %d mapping rows to %s", len(mapping), args.out)


if __name__ == "__main__":
    main()
