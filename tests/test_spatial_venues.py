"""Tests for spatial.venues — per-station nearest-venue distance/count.

Hermetic: 3 synthetic stations × 2 synthetic venues with hand-computed
expected distances. No real data, no network.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from fuel_pred.spatial.venues import (
    DEFAULT_RADIUS_KM,
    EARTH_RADIUS_KM,
    _haversine_km_matrix,
    compute_station_venues,
)


def _haversine_one(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Single-pair haversine in km — used for hand-calculated expectations."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _venues_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # 'big' near Sydney CBD; 'small' ~5 km north of 'big'
            {
                "venue_id": "big",
                "name": "Big Stadium",
                "lat": -33.870,
                "lon": 151.210,
                "capacity": 80000,
                "type": "stadium",
            },
            {
                "venue_id": "small",
                "name": "Small Arena",
                "lat": -33.825,  # ~5 km north of 'big'
                "lon": 151.210,
                "capacity": 12000,
                "type": "entertainment_centre",
            },
        ]
    )


def _stations_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # s_close: literally on top of 'big' — should round to ~0 km, nearest = big
            {"station_id": "s_close", "lat": -33.870, "lon": 151.210},
            # s_between: roughly halfway between big and small; near both
            {"station_id": "s_between", "lat": -33.8475, "lon": 151.210},
            # s_far: well south of both — nearest = big, well outside radius
            {"station_id": "s_far", "lat": -34.500, "lon": 150.800},
        ]
    )


def test_nearest_venue_is_closest_and_distance_is_correct() -> None:
    stations = _stations_fixture()
    venues = _venues_fixture()
    out = compute_station_venues(stations, venues, radius_km=DEFAULT_RADIUS_KM)

    # Schema sanity.
    expected_cols = {
        "station_id",
        "stn_nearest_venue_km",
        "stn_nearest_venue_id",
        "stn_nearest_venue_capacity",
        "stn_nearest_venue_type",
        "stn_n_venues_within_5km",
    }
    assert expected_cols.issubset(out.columns), (
        f"output missing columns: {expected_cols - set(out.columns)}"
    )

    # s_close sits on top of 'big' — distance ~0, nearest = big.
    close = out.set_index("station_id").loc["s_close"]
    assert close["stn_nearest_venue_id"] == "big"
    assert close["stn_nearest_venue_capacity"] == pytest.approx(80000)
    assert close["stn_nearest_venue_type"] == "stadium"
    assert close["stn_nearest_venue_km"] < 0.01, (
        f"s_close should be ~0 km from big, got {close['stn_nearest_venue_km']}"
    )

    # s_far — nearest = big; sanity-check the actual distance against the
    # single-pair haversine impl.
    far = out.set_index("station_id").loc["s_far"]
    assert far["stn_nearest_venue_id"] == "big"
    expected_far_big = _haversine_one(-34.500, 150.800, -33.870, 151.210)
    assert far["stn_nearest_venue_km"] == pytest.approx(expected_far_big, rel=1e-6)


def test_radius_count_is_correct() -> None:
    """Radius-count column counts venues within the radius (inclusive).

    Column name is parameterised by the radius (e.g. ``stn_n_venues_within_5km``
    at the default 5 km radius), so test under both the default and a
    tighter custom radius.
    """
    stations = _stations_fixture()
    venues = _venues_fixture()
    out = compute_station_venues(stations, venues, radius_km=5.0).set_index("station_id")

    # s_far is hundreds of km from both → count must be 0 under the default
    # 5 km radius column.
    assert out.loc["s_far", "stn_n_venues_within_5km"] == 0

    # s_close sits on top of 'big' (~0 km) and ~5 km from 'small'. With a
    # 4 km radius the column name changes (`stn_n_venues_within_4km`) and
    # the count must be exactly 1 (big alone).
    out_strict = compute_station_venues(stations, venues, radius_km=4.0).set_index(
        "station_id"
    )
    assert out_strict.loc["s_close", "stn_n_venues_within_4km"] == 1, (
        "s_close should have exactly 1 venue within 4 km"
    )


def test_haversine_matrix_shape_and_values() -> None:
    """Vectorised matrix produces same answer as the single-pair scalar fn."""
    import numpy as np

    s_lats = np.array([-33.870, -34.500])
    s_lons = np.array([151.210, 150.800])
    v_lats = np.array([-33.870, -33.825])
    v_lons = np.array([151.210, 151.210])

    m = _haversine_km_matrix(s_lats, s_lons, v_lats, v_lons)
    assert m.shape == (2, 2)
    # (s_close → big) should be ~0
    assert m[0, 0] < 0.01
    # (s_far → big) should agree with the scalar impl
    assert m[1, 0] == pytest.approx(
        _haversine_one(-34.500, 150.800, -33.870, 151.210), rel=1e-6
    )


def test_stations_without_coords_get_null_venue_features() -> None:
    """Stations with NaN lat/lon shouldn't crash; they get null/zero."""
    import numpy as np

    stations = pd.DataFrame(
        [
            {"station_id": "s_ok", "lat": -33.870, "lon": 151.210},
            {"station_id": "s_no_coords", "lat": np.nan, "lon": np.nan},
        ]
    )
    venues = _venues_fixture()
    out = compute_station_venues(stations, venues).set_index("station_id")

    assert pd.isna(out.loc["s_no_coords", "stn_nearest_venue_km"])
    assert pd.isna(out.loc["s_no_coords", "stn_nearest_venue_capacity"])
    assert out.loc["s_no_coords", "stn_n_venues_within_5km"] == 0
    # The other station works fine.
    assert out.loc["s_ok", "stn_nearest_venue_km"] < 0.01


def test_io_roundtrip_via_paths(tmp_path: Path) -> None:
    """End-to-end CLI helper: write stations + venues, run, read back."""
    from fuel_pred.spatial.venues import compute_station_venues_from_paths

    stations_path = tmp_path / "stations.parquet"
    venues_path = tmp_path / "venues.csv"
    out_path = tmp_path / "stations_venues.parquet"

    _stations_fixture().to_parquet(stations_path, index=False)
    _venues_fixture().to_csv(venues_path, index=False)

    compute_station_venues_from_paths(stations_path, venues_path, out_path)
    assert out_path.exists()
    result = pd.read_parquet(out_path).set_index("station_id")
    assert result.loc["s_close", "stn_nearest_venue_id"] == "big"
    assert result.loc["s_close", "stn_nearest_venue_km"] < 0.01
