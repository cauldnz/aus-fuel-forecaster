"""Hermetic tests for spatial.gfs_grid — station→grid mapping for 3 GFS/GEFS resolutions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fuel_pred.spatial import gfs_grid

# ----------------------------- _lat_lon_axes -----------------------------


def test_axes_025() -> None:
    lats, lons = gfs_grid._lat_lon_axes(0.25)
    # 180 / 0.25 + 1 = 721 lats; 360 / 0.25 = 1440 lons.
    assert lats.shape == (721,)
    assert lons.shape == (1440,)
    assert lats[0] == 90.0
    assert lats[-1] == -90.0
    assert lons[0] == 0.0
    assert lons[-1] == pytest.approx(359.75)


def test_axes_1deg() -> None:
    lats, lons = gfs_grid._lat_lon_axes(1.0)
    assert lats.shape == (181,)
    assert lons.shape == (360,)


def test_axes_05deg() -> None:
    lats, lons = gfs_grid._lat_lon_axes(0.5)
    assert lats.shape == (361,)
    assert lons.shape == (720,)


# ----------------------------- _bilinear_for_station -----------------------------


def test_bilinear_on_grid_point_collapses_to_nearest() -> None:
    """Station exactly on a grid point: one weight is 1, others are 0."""
    # 0.25° grid, station at lat=-33.75, lon=151.25 (exactly on a grid point).
    (
        nearest_lat, nearest_lon,
        lat0, _lat1, lon0, _lon1,
        w00, w01, w10, w11,
    ) = gfs_grid._bilinear_for_station(-33.75, 151.25, 0.25)
    assert (w00, w01, w10, w11) == (1.0, 0.0, 0.0, 0.0)
    assert (nearest_lat, nearest_lon) == (lat0, lon0)
    # lat=-33.75 → idx = (90 - (-33.75)) / 0.25 = 495
    assert lat0 == 495
    # lon=151.25 → idx = 605
    assert lon0 == 605


def test_bilinear_off_grid_weights_sum_to_one() -> None:
    """Station between grid points: 4 non-zero weights summing to 1.0."""
    # 0.25° grid, station at lat=-33.87, lon=151.21 (Sydney CBD).
    (
        _nearest_lat, _nearest_lon,
        _lat0, _lat1, _lon0, _lon1,
        w00, w01, w10, w11,
    ) = gfs_grid._bilinear_for_station(-33.87, 151.21, 0.25)
    assert sum([w00, w01, w10, w11]) == pytest.approx(1.0)
    # All four weights should be positive and < 1.
    assert all(0 < w < 1 for w in [w00, w01, w10, w11])


def test_bilinear_lat_descending_alpha() -> None:
    """alpha_lat reflects descending-lat convention.

    At 1° grid: lat=-33.5 sits between grid points lat[123]=-33 (idx 123)
    and lat[124]=-34 (idx 124). Since lats descend, the station is 0.5 of
    the way from lat[123] to lat[124], so alpha_lat=0.5.

    Weights: w_00 (top, west) = 0.5 * (1 - alpha_lon)
             w_10 (bottom, west) = 0.5 * (1 - alpha_lon)
    Both halves carry equal weight, regardless of lon.
    """
    # Pick lon exactly on a grid point so alpha_lon = 0.
    (_, _, lat0, lat1, _, _, w00, w01, w10, w11) = gfs_grid._bilinear_for_station(
        -33.5, 151.0, 1.0,
    )
    assert lat0 == 123  # 90 - (-33) = 123
    assert lat1 == 124
    # alpha_lon == 0 → w_01 and w_11 are 0.
    assert (w01, w11) == (0.0, 0.0)
    # alpha_lat == 0.5 → w_00 == w_10 == 0.5
    assert (w00, w10) == (0.5, 0.5)


def test_bilinear_lon_eastern_hemisphere() -> None:
    """NSW lons (140..154) all sit in positive-lon-only branch (no wrap)."""
    _, _, _, _, lon0, lon1, *_ = gfs_grid._bilinear_for_station(-33.87, 151.21, 0.25)
    # 151.21 / 0.25 = 604.84 → lon0 = 604, lon1 = 605
    assert lon0 == 604
    assert lon1 == 605


def test_bilinear_negative_lon_normalises_to_eastern() -> None:
    """A western-hemisphere station (negative lon) is folded into 0..360."""
    # lon=-179 == 181 in 0..360.
    _, _, _, _, lon0, lon1, *_ = gfs_grid._bilinear_for_station(-33.0, -179.0, 1.0)
    assert lon0 == 181  # 181/1 = 181
    assert lon1 == 182


def test_bilinear_rejects_out_of_range_lat() -> None:
    with pytest.raises(ValueError, match="lat"):
        gfs_grid._bilinear_for_station(95.0, 151.0, 0.25)


def test_bilinear_clamps_at_south_pole() -> None:
    """A station at exactly -90: lat indices clamp to the last grid row."""
    (
        nearest_lat, _,
        lat0, lat1, _, _,
        w00, w01, w10, w11,
    ) = gfs_grid._bilinear_for_station(-90.0, 151.0, 1.0)
    # n_lat = 181, last index = 180
    assert lat0 == 180
    assert lat1 == 180
    assert nearest_lat == 180
    # Weights still sum to 1 (alpha_lat clamped to 0 ⇒ w_10/11 = 0).
    assert sum([w00, w01, w10, w11]) == pytest.approx(1.0)


# ----------------------------- compute_station_grid_mapping -----------------------------


def _five_stations() -> pd.DataFrame:
    """Five fixture stations covering NSW + an outlier with null lat/lon."""
    return pd.DataFrame(
        [
            # Sydney CBD
            {"station_id": "syd_cbd", "lat": -33.87, "lon": 151.21},
            # Newcastle
            {"station_id": "newcastle", "lat": -32.93, "lon": 151.78},
            # Wagga Wagga (regional NSW)
            {"station_id": "wagga", "lat": -35.12, "lon": 147.37},
            # Tweed Heads (NSW north border)
            {"station_id": "tweed", "lat": -28.18, "lon": 153.55},
            # Station with no lat/lon — should be dropped.
            {"station_id": "ghost", "lat": None, "lon": None},
        ]
    )


def test_mapping_outputs_one_row_per_usable_station() -> None:
    mapping = gfs_grid.compute_station_grid_mapping(_five_stations())
    assert len(mapping) == 4  # ghost dropped
    assert set(mapping["station_id"]) == {"syd_cbd", "newcastle", "wagga", "tweed"}


def test_mapping_has_all_three_resolutions() -> None:
    mapping = gfs_grid.compute_station_grid_mapping(_five_stations())
    for prefix in ("gfs", "gefs05", "gefs1"):
        for suffix in (
            "_lat_idx", "_lon_idx",
            "_bl_lat_idx_0", "_bl_lat_idx_1",
            "_bl_lon_idx_0", "_bl_lon_idx_1",
            "_bl_w_00", "_bl_w_01", "_bl_w_10", "_bl_w_11",
        ):
            col = prefix + suffix
            assert col in mapping.columns, f"missing column {col}"


def test_mapping_weights_sum_to_one_for_every_station() -> None:
    mapping = gfs_grid.compute_station_grid_mapping(_five_stations())
    for prefix in ("gfs", "gefs05", "gefs1"):
        w_cols = [f"{prefix}_bl_w_{ij}" for ij in ("00", "01", "10", "11")]
        row_sums = mapping[w_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.to_numpy(), 1.0, atol=1e-12)


def test_mapping_indices_within_grid_bounds() -> None:
    mapping = gfs_grid.compute_station_grid_mapping(_five_stations())
    # GFS 0.25°: lat in [0, 720], lon in [0, 1439]
    assert (mapping["gfs_lat_idx"] >= 0).all()
    assert (mapping["gfs_lat_idx"] <= 720).all()
    assert (mapping["gfs_lon_idx"] >= 0).all()
    assert (mapping["gfs_lon_idx"] <= 1439).all()
    # GEFS 1°: lat in [0, 180], lon in [0, 359]
    assert (mapping["gefs1_lat_idx"] <= 180).all()
    assert (mapping["gefs1_lon_idx"] <= 359).all()


def test_mapping_raises_on_missing_columns() -> None:
    bad = pd.DataFrame({"station_id": ["s1"], "lat": [-33.0]})  # no lon
    with pytest.raises(ValueError, match="missing required columns"):
        gfs_grid.compute_station_grid_mapping(bad)


def test_mapping_sydney_resolves_to_expected_grid_cell() -> None:
    """Sydney CBD (-33.87, 151.21) at 0.25° should bracket lat[495..496] = -33.75..-34.0
    and lon[604..605] = 151.0..151.25."""
    mapping = gfs_grid.compute_station_grid_mapping(_five_stations())
    syd = mapping.set_index("station_id").loc["syd_cbd"]
    assert int(syd["gfs_bl_lat_idx_0"]) == 495
    assert int(syd["gfs_bl_lat_idx_1"]) == 496
    assert int(syd["gfs_bl_lon_idx_0"]) == 604
    assert int(syd["gfs_bl_lon_idx_1"]) == 605


# ----------------------------- CLI / atomic write -----------------------------


def test_cli_writes_atomic_no_tmp_remaining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stations_path = tmp_path / "stations.parquet"
    _five_stations().to_parquet(stations_path, engine="pyarrow", compression="zstd", index=False)
    out_path = tmp_path / "mapping.parquet"

    monkeypatch.setattr(
        "sys.argv",
        ["gfs_grid", "--stations", str(stations_path), "--out", str(out_path)],
    )
    gfs_grid.main()

    assert out_path.exists()
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == [], f"unexpected .tmp leftover: {tmps}"
    df = pd.read_parquet(out_path)
    assert len(df) == 4
