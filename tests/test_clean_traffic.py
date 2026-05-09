"""Hermetic tests for clean.traffic."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fuel_pred.clean import traffic as ct


def _stations(rows: list[dict[str, object]]) -> pd.DataFrame:
    base_keys = {
        "station_key": None,
        "station_id": None,
        "name": "Test",
        "road_name": "Some Rd",
        "suburb": "Sydney",
        "post_code": "2000",
        "lga": "Sydney",
        "rms_region": "Sydney",
        "wgs84_latitude": -33.8,
        "wgs84_longitude": 151.2,
        "quality_rating": "5",
        "permanent_station": "1",
        "road_functional_hierarchy": "Arterial",
    }
    return pd.DataFrame([{**base_keys, **r} for r in rows])


def _hourly(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {
        "station_key": None,
        "date": None,
        "traffic_direction_seq": "0",
        "classification_seq": "0",
        "daily_total": 0,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _setup(tmp_path: Path, stations: pd.DataFrame, hourly: pd.DataFrame) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    stations.to_parquet(raw / "stations.parquet", engine="pyarrow", compression="zstd", index=False)
    hourly.to_parquet(raw / "hourly.parquet", engine="pyarrow", compression="zstd", index=False)
    return raw


def test_drops_non_permanent_and_low_quality_stations(tmp_path: Path) -> None:
    stations = _stations(
        [
            {"station_key": "good", "quality_rating": "5", "permanent_station": "1"},
            {"station_key": "lowqual", "quality_rating": "2", "permanent_station": "1"},
            {"station_key": "casual", "quality_rating": "5", "permanent_station": "0"},
            {"station_key": "nolatlon", "quality_rating": "5", "permanent_station": "1",
             "wgs84_latitude": None, "wgs84_longitude": None},
        ]
    )
    hourly = _hourly([{"station_key": "good", "date": "2024-08-01", "daily_total": 100}])
    raw = _setup(tmp_path, stations, hourly)

    ct.clean(raw, tmp_path / "daily.parquet", tmp_path / "stations.parquet")

    out_stations = pd.read_parquet(tmp_path / "stations.parquet")
    assert list(out_stations["station_key"]) == ["good"]


def test_aggregates_breakdown_classification(tmp_path: Path) -> None:
    """Classification 2 (light) + 3 (heavy) re-aggregate to total — verified the
    common per-station scheme. Sums also handle the directional split."""
    stations = _stations([{"station_key": "S1"}])
    hourly = _hourly(
        [
            {"station_key": "S1", "date": "2024-08-01", "traffic_direction_seq": "0",
             "classification_seq": "2", "daily_total": 100},
            {"station_key": "S1", "date": "2024-08-01", "traffic_direction_seq": "1",
             "classification_seq": "2", "daily_total": 80},
            {"station_key": "S1", "date": "2024-08-01", "traffic_direction_seq": "0",
             "classification_seq": "3", "daily_total": 60},
            {"station_key": "S1", "date": "2024-08-01", "traffic_direction_seq": "1",
             "classification_seq": "3", "daily_total": 40},
        ]
    )
    raw = _setup(tmp_path, stations, hourly)

    ct.clean(raw, tmp_path / "daily.parquet", tmp_path / "stations.parquet")

    daily = pd.read_parquet(tmp_path / "daily.parquet")
    assert len(daily) == 1
    # All four rows summed: 100 + 80 + 60 + 40 = 280.
    assert daily["daily_total"].iloc[0] == 280


def test_aggregates_pretotalled_classification(tmp_path: Path) -> None:
    """A station that emits only `classification_seq=0` (a pre-totalled row)
    yields the same answer as the breakdown scheme."""
    stations = _stations([{"station_key": "S1"}])
    hourly = _hourly(
        [
            {"station_key": "S1", "date": "2024-08-01", "traffic_direction_seq": "0",
             "classification_seq": "0", "daily_total": 180},
            {"station_key": "S1", "date": "2024-08-01", "traffic_direction_seq": "1",
             "classification_seq": "0", "daily_total": 100},
        ]
    )
    raw = _setup(tmp_path, stations, hourly)

    ct.clean(raw, tmp_path / "daily.parquet", tmp_path / "stations.parquet")
    daily = pd.read_parquet(tmp_path / "daily.parquet")
    assert daily["daily_total"].iloc[0] == 280


def test_drops_hourly_rows_for_filtered_stations(tmp_path: Path) -> None:
    stations = _stations(
        [
            {"station_key": "good"},
            {"station_key": "bad", "quality_rating": "1"},
        ]
    )
    hourly = _hourly(
        [
            {"station_key": "good", "date": "2024-08-01", "daily_total": 100},
            {"station_key": "bad", "date": "2024-08-01", "daily_total": 999},
        ]
    )
    raw = _setup(tmp_path, stations, hourly)

    ct.clean(raw, tmp_path / "daily.parquet", tmp_path / "stations.parquet")
    daily = pd.read_parquet(tmp_path / "daily.parquet")
    assert list(daily["station_key"]) == ["good"]


def test_missing_input_raises(tmp_path: Path) -> None:
    raw = tmp_path / "empty"
    raw.mkdir()
    with pytest.raises(RuntimeError, match="missing"):
        ct.clean(raw, tmp_path / "daily.parquet", tmp_path / "stations.parquet")


def test_hourly_missing_required_columns_raises(tmp_path: Path) -> None:
    stations = _stations([{"station_key": "good"}])
    hourly = pd.DataFrame(
        [{"station_key": "good", "date": "2024-08-01"}]  # no daily_total
    )
    raw = _setup(tmp_path, stations, hourly)
    with pytest.raises(RuntimeError, match="daily_total"):
        ct.clean(raw, tmp_path / "daily.parquet", tmp_path / "stations.parquet")


def test_handles_tz_aware_dates(tmp_path: Path) -> None:
    """Real TfNSW dates are tz-aware UTC; cleaner must normalise."""
    stations = _stations([{"station_key": "S1"}])
    hourly = _hourly(
        [
            {"station_key": "S1", "date": "2024-08-01T00:00:00+00:00", "daily_total": 100},
            {"station_key": "S1", "date": "2024-08-02T00:00:00+00:00", "daily_total": 200},
        ]
    )
    raw = _setup(tmp_path, stations, hourly)

    ct.clean(raw, tmp_path / "daily.parquet", tmp_path / "stations.parquet")
    daily = pd.read_parquet(tmp_path / "daily.parquet")
    assert len(daily) == 2
    import datetime as dt

    assert daily["date"].iloc[0] == dt.date(2024, 8, 1)
