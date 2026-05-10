"""Hermetic tests for spatial.nearest."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fuel_pred.spatial import nearest as nn


def _stations(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _counters(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ----------------------------- haversine sanity -----------------------------


def test_haversine_known_pairs() -> None:
    """A few hand-checked great-circle distances."""
    # Sydney CBD (-33.87, 151.21) to Melbourne CBD (-37.81, 144.96) ≈ 714 km.
    syd_mel = nn._haversine_km(-33.87, 151.21, -37.81, 144.96)
    assert 700 < syd_mel < 730

    # Identity → 0.
    assert nn._haversine_km(-33.87, 151.21, -33.87, 151.21) == pytest.approx(0.0)

    # Same latitude, 1 degree of longitude apart at -34° lat ≈ 92 km.
    one_deg = nn._haversine_km(-34.0, 150.0, -34.0, 151.0)
    assert 90 < one_deg < 95


# ----------------------------- compute_top_n: shape -----------------------------


@pytest.fixture
def small_setup() -> tuple[pd.DataFrame, pd.DataFrame]:
    stations = _stations(
        [
            {"station_id": "s1", "lat": -33.93, "lon": 151.20},  # Mascot-ish
            {"station_id": "s2", "lat": -33.65, "lon": 151.32},  # Newport-ish
            {"station_id": "s3", "lat": -32.93, "lon": 151.78},  # Newcastle-ish
        ]
    )
    counters = _counters(
        [
            {"station_key": "C-MASCOT", "wgs84_latitude": -33.93, "wgs84_longitude": 151.20},
            {"station_key": "C-MASCOT-EAST", "wgs84_latitude": -33.93, "wgs84_longitude": 151.21},
            {"station_key": "C-NEWPORT", "wgs84_latitude": -33.65, "wgs84_longitude": 151.32},
            {"station_key": "C-NEWCASTLE", "wgs84_latitude": -32.93, "wgs84_longitude": 151.78},
            {"station_key": "C-SYDCBD", "wgs84_latitude": -33.87, "wgs84_longitude": 151.21},
            {"station_key": "C-WOLLONGONG", "wgs84_latitude": -34.42, "wgs84_longitude": 150.89},
        ]
    )
    return stations, counters


def test_top_table_has_expected_schema(small_setup: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    stations, counters = small_setup
    top, _ = nn.compute_top_n(stations, counters, top_n=3)
    assert list(top.columns) == ["station_id", "counter_rank", "counter_id", "distance_km"]
    # 3 stations x top-3 = 9 rows.
    assert len(top) == 9
    # Ranks per station are 1..3 in order.
    for sid in ("s1", "s2", "s3"):
        ranks = top.loc[top["station_id"] == sid, "counter_rank"].tolist()
        assert ranks == [1, 2, 3]
    # distance_km is non-decreasing within each station.
    for sid in ("s1", "s2", "s3"):
        ds = top.loc[top["station_id"] == sid, "distance_km"].tolist()
        assert ds == sorted(ds)


def test_summary_has_expected_schema(small_setup: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    stations, counters = small_setup
    _, summary = nn.compute_top_n(stations, counters, top_n=3, radius_km=5.0)
    expected = {"station_id", "stn_distance_to_sydney_terminal_km", "stn_n_counters_within_5km"}
    assert expected <= set(summary.columns)
    assert len(summary) == 3


# ----------------------------- compute_top_n: correctness -----------------------------


def test_nearest_counter_is_co_located(small_setup: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """When a counter is at exactly the same lat/lon as the station,
    it must be rank 1 with distance ≈ 0."""
    stations, counters = small_setup
    top, _ = nn.compute_top_n(stations, counters, top_n=3)
    s1_top = top[top["station_id"] == "s1"].sort_values("counter_rank")
    # C-MASCOT is at (-33.93, 151.20) — exactly Mascot fixture.
    assert s1_top.iloc[0]["counter_id"] == "C-MASCOT"
    assert s1_top.iloc[0]["distance_km"] == pytest.approx(0.0, abs=1e-3)


def test_terminal_distance_from_botany_anchor(
    small_setup: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """Botany terminal at (-33.9619, 151.2095). Mascot is ~3.5 km north of it."""
    stations, counters = small_setup
    _, summary = nn.compute_top_n(stations, counters, top_n=3)
    s1_terminal = summary.loc[summary["station_id"] == "s1", "stn_distance_to_sydney_terminal_km"]
    # Mascot to Botany — about 3.6 km.
    assert 2.0 < float(s1_terminal.iloc[0]) < 5.0
    # Newcastle is ~120 km north of Botany.
    s3_terminal = summary.loc[summary["station_id"] == "s3", "stn_distance_to_sydney_terminal_km"]
    assert 110 < float(s3_terminal.iloc[0]) < 130


def test_radius_count_within_5km(small_setup: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """s1 (Mascot) has 2 counters within 5 km (C-MASCOT, C-MASCOT-EAST,
    plus C-SYDCBD ~7 km away). Verify the count."""
    stations, counters = small_setup
    _, summary = nn.compute_top_n(stations, counters, top_n=3, radius_km=5.0)
    s1_count = int(summary.loc[summary["station_id"] == "s1", "stn_n_counters_within_5km"].iloc[0])
    # Mascot itself + Mascot-East within 5km of Mascot. CBD is ~7km away.
    assert s1_count == 2


# ----------------------------- compute_top_n: edge cases -----------------------------


def test_station_with_missing_lat_lon_omitted_from_top_table_kept_in_summary() -> None:
    """A station with NaN lat/lon shouldn't appear in top-N table but must
    still appear in the per-station summary (with null distance, count=0)."""
    stations = _stations(
        [
            {"station_id": "s_ok", "lat": -33.93, "lon": 151.20},
            {"station_id": "s_missing", "lat": np.nan, "lon": np.nan},
        ]
    )
    counters = _counters(
        [{"station_key": "C1", "wgs84_latitude": -33.93, "wgs84_longitude": 151.20}]
    )
    top, summary = nn.compute_top_n(stations, counters, top_n=2)
    # top table only has s_ok rows.
    assert set(top["station_id"]) == {"s_ok"}
    # summary has both stations.
    assert set(summary["station_id"]) == {"s_ok", "s_missing"}
    s_missing_row = summary.loc[summary["station_id"] == "s_missing"].iloc[0]
    assert pd.isna(s_missing_row["stn_distance_to_sydney_terminal_km"])
    assert int(s_missing_row["stn_n_counters_within_5km"]) == 0


def test_top_n_capped_at_counter_pool_size() -> None:
    """If the counter pool is smaller than top_n, the per-station rank
    list must shrink — never crash, never emit ranks > pool size."""
    stations = _stations([{"station_id": "s1", "lat": -33.93, "lon": 151.20}])
    counters = _counters(
        [
            {"station_key": "C1", "wgs84_latitude": -33.93, "wgs84_longitude": 151.20},
            {"station_key": "C2", "wgs84_latitude": -33.94, "wgs84_longitude": 151.21},
        ]
    )
    top, _ = nn.compute_top_n(stations, counters, top_n=10)
    assert sorted(top["counter_rank"]) == [1, 2]


def test_no_counters_with_lat_lon_raises() -> None:
    stations = _stations([{"station_id": "s1", "lat": -33.93, "lon": 151.20}])
    counters = _counters(
        [{"station_key": "C-broken", "wgs84_latitude": np.nan, "wgs84_longitude": np.nan}]
    )
    with pytest.raises(RuntimeError, match="no traffic counters"):
        nn.compute_top_n(stations, counters, top_n=3)


def test_no_stations_with_lat_lon_returns_empty_top(
    small_setup: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """All stations missing lat/lon → empty top, summary still has all stations."""
    _, counters = small_setup
    stations = _stations(
        [
            {"station_id": "s1", "lat": np.nan, "lon": np.nan},
            {"station_id": "s2", "lat": np.nan, "lon": np.nan},
        ]
    )
    top, summary = nn.compute_top_n(stations, counters, top_n=3)
    assert len(top) == 0
    assert sorted(summary["station_id"]) == ["s1", "s2"]


def test_missing_required_columns_raises() -> None:
    bad_stations = _stations([{"name": "X", "lat": -33.0, "lon": 151.0}])
    counters = _counters(
        [{"station_key": "C1", "wgs84_latitude": -33.0, "wgs84_longitude": 151.0}]
    )
    with pytest.raises(ValueError, match="station_id"):
        nn.compute_top_n(bad_stations, counters)

    stations = _stations([{"station_id": "s1", "lat": -33.0, "lon": 151.0}])
    bad_counters = _counters(
        [{"name": "X", "wgs84_latitude": -33.0, "wgs84_longitude": 151.0}]
    )
    with pytest.raises(ValueError, match="station_key"):
        nn.compute_top_n(stations, bad_counters)


# ----------------------------- compute_nearest (file IO) -----------------------------


def test_compute_nearest_writes_two_parquets(
    tmp_path: Path, small_setup: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    stations, counters = small_setup
    s_path = tmp_path / "stations.parquet"
    c_path = tmp_path / "counters.parquet"
    out = tmp_path / "station_to_counter.parquet"
    summary_out = tmp_path / "summary.parquet"
    stations.to_parquet(s_path, engine="pyarrow", compression="zstd", index=False)
    counters.to_parquet(c_path, engine="pyarrow", compression="zstd", index=False)

    nn.compute_nearest(s_path, c_path, out, summary_out=summary_out, top_n=2)

    top = pd.read_parquet(out)
    assert len(top) == 6  # 3 stations x 2 ranks
    summary = pd.read_parquet(summary_out)
    assert "stn_distance_to_sydney_terminal_km" in summary.columns


def test_compute_nearest_default_summary_path_beside_out(
    tmp_path: Path, small_setup: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    stations, counters = small_setup
    s_path = tmp_path / "stations.parquet"
    c_path = tmp_path / "counters.parquet"
    out = tmp_path / "nearest.parquet"
    stations.to_parquet(s_path, engine="pyarrow", compression="zstd", index=False)
    counters.to_parquet(c_path, engine="pyarrow", compression="zstd", index=False)

    nn.compute_nearest(s_path, c_path, out, top_n=2)

    expected_summary = tmp_path / "nearest_summary.parquet"
    assert expected_summary.exists()
