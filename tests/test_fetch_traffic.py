"""Hermetic tests for fetch.traffic.

CKAN endpoints + the ZIP download for the hourly resource are mocked with
`responses`. The hourly ZIP body is constructed in-memory.
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import responses

from fuel_pred.fetch import traffic


def _package_show_payload(resources: list[dict[str, Any]]) -> dict[str, Any]:
    return {"success": True, "result": {"resources": resources}}


def _datastore_payload(records: list[dict[str, Any]], total: int) -> dict[str, Any]:
    return {"success": True, "result": {"records": records, "total": total}}


def _zip_with_csvs(name_to_csv: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in name_to_csv.items():
            zf.writestr(name, body)
    return buf.getvalue()


HOURLY_ZIP_URL = "https://example.com/hourly.zip"


@pytest.fixture
def package_resources() -> list[dict[str, Any]]:
    return [
        {
            "id": "stations-id",
            "name": "Road Traffic Counts Station Reference (API Generated CSV)",
            "format": "CSV",
            "datastore_active": True,
            "url": "https://example.com/stations.csv",
        },
        {
            "id": "hourly-id",
            "name": "Road Traffic Counts Hourly Permanent (API Generated CSVs)",
            "format": "ZIP",
            "datastore_active": False,
            "url": HOURLY_ZIP_URL,
        },
        {
            "id": "extra-id",
            "name": "Some other resource",
            "format": "CSV",
            "datastore_active": False,
            "url": "https://example.com/other.csv",
        },
    ]


@pytest.fixture
def hourly_zip() -> bytes:
    csv_a = (
        "station_id,date,hour,count\n"
        "S1,2024-01-01,0,12\n"
        "S1,2024-01-01,1,8\n"
        "S2,2023-06-01,10,50\n"  # out of range
    )
    csv_b = (
        "station_id,date,hour,count\n"
        "S2,2024-02-15,9,220\n"
    )
    return _zip_with_csvs({"part_a.csv": csv_a, "part_b.csv": csv_b, "README.txt": "ignored"})


def _register_full_fetch(
    package_resources: list[dict[str, Any]], hourly_zip: bytes, *, page_size: int = 2
) -> None:
    """Register the package_show + stations datastore + hourly ZIP responses."""
    responses.add(
        responses.GET,
        f"{traffic.API_ROOT}/package_show",
        json=_package_show_payload(package_resources),
        status=200,
    )
    stations = [
        {"station_id": "S1", "lat": -33.8, "lon": 151.2, "suburb": "Sydney"},
        {"station_id": "S2", "lat": -32.9, "lon": 151.7, "suburb": "Newcastle"},
    ]
    responses.add(
        responses.GET,
        f"{traffic.API_ROOT}/datastore_search",
        json=_datastore_payload(stations, total=2),
        status=200,
        match=[responses.matchers.query_param_matcher(
            {"resource_id": "stations-id", "limit": str(page_size), "offset": "0"}
        )],
    )
    responses.add(responses.GET, HOURLY_ZIP_URL, body=hourly_zip, status=200)


@responses.activate
def test_writes_two_parquets_with_filtered_hourly(
    tmp_path: Path,
    package_resources: list[dict[str, Any]],
    hourly_zip: bytes,
) -> None:
    _register_full_fetch(package_resources, hourly_zip)

    traffic.fetch("2024-01-01", "2024-12-31", tmp_path, page_size=2)

    stations = pd.read_parquet(tmp_path / "stations.parquet")
    hourly = pd.read_parquet(tmp_path / "hourly.parquet")

    assert len(stations) == 2
    assert {"station_id", "lat", "lon"}.issubset(stations.columns)

    # Two CSVs concat'd; out-of-range S2 row dropped → 3 in-range rows.
    assert len(hourly) == 3
    assert "2023" not in "".join(str(d) for d in hourly["date"].astype(str))


@responses.activate
def test_skips_when_cache_fresh(tmp_path: Path) -> None:
    (tmp_path / "stations.parquet").write_bytes(b"placeholder-stations")
    (tmp_path / "hourly.parquet").write_bytes(b"placeholder-hourly")

    # No CKAN mocks registered — any request would explode.
    traffic.fetch("2024-01-01", "2024-12-31", tmp_path, max_age_days=1.0)

    assert (tmp_path / "stations.parquet").read_bytes() == b"placeholder-stations"
    assert (tmp_path / "hourly.parquet").read_bytes() == b"placeholder-hourly"


@responses.activate
def test_force_bypasses_cache(
    tmp_path: Path,
    package_resources: list[dict[str, Any]],
    hourly_zip: bytes,
) -> None:
    (tmp_path / "stations.parquet").write_bytes(b"placeholder-stations")
    (tmp_path / "hourly.parquet").write_bytes(b"placeholder-hourly")
    _register_full_fetch(package_resources, hourly_zip)

    traffic.fetch("2024-01-01", "2024-12-31", tmp_path, force=True, page_size=2)

    stations = pd.read_parquet(tmp_path / "stations.parquet")
    assert len(stations) == 2


@responses.activate
def test_stale_cache_triggers_refetch(
    tmp_path: Path,
    package_resources: list[dict[str, Any]],
    hourly_zip: bytes,
) -> None:
    import os

    (tmp_path / "stations.parquet").write_bytes(b"placeholder")
    (tmp_path / "hourly.parquet").write_bytes(b"placeholder")
    old = time.time() - 5 * 86400
    os.utime(tmp_path / "stations.parquet", (old, old))
    os.utime(tmp_path / "hourly.parquet", (old, old))

    _register_full_fetch(package_resources, hourly_zip)

    traffic.fetch("2024-01-01", "2024-12-31", tmp_path, max_age_days=1.0, page_size=2)

    hourly = pd.read_parquet(tmp_path / "hourly.parquet")
    assert len(hourly) == 3


@responses.activate
def test_resource_lookup_raises_when_missing(tmp_path: Path) -> None:
    responses.add(
        responses.GET,
        f"{traffic.API_ROOT}/package_show",
        json=_package_show_payload(
            [{"id": "x", "name": "Annual report"}]
        ),
        status=200,
    )
    with pytest.raises(RuntimeError, match="station reference"):
        traffic.fetch("2024-01-01", "2024-12-31", tmp_path, page_size=2)


def test_find_resource_id_substring_match() -> None:
    resources = [
        {"id": "a", "name": "Road Traffic Counts Hourly Permanent (API Generated CSVs)"},
        {"id": "b", "name": "Other"},
    ]
    rid = traffic._find_resource_id(resources, traffic.HOURLY_RESOURCE_HINT)
    assert rid == "a"


@responses.activate
def test_ckan_error_payload_raises(tmp_path: Path) -> None:
    responses.add(
        responses.GET,
        f"{traffic.API_ROOT}/package_show",
        json={"success": False, "error": {"message": "not found"}},
        status=200,
    )
    with pytest.raises(RuntimeError, match="CKAN error"):
        traffic.fetch("2024-01-01", "2024-12-31", tmp_path)


def test_no_real_network() -> None:
    """The pure helpers should work without any sockets."""
    assert traffic._find_resource_id(
        [{"id": "x", "name": "Road Traffic Counts Hourly Permanent (zip)"}],
        traffic.HOURLY_RESOURCE_HINT,
    ) == "x"


@responses.activate
def test_unknown_resource_format_raises(
    tmp_path: Path,
) -> None:
    """A resource with no datastore and an unrecognised format errors out."""
    resources = [
        {
            "id": "stations-id",
            "name": "Road Traffic Counts Station Reference",
            "format": "CSV",
            "datastore_active": True,
            "url": "https://example.com/stations.csv",
        },
        {
            "id": "hourly-id",
            "name": "Road Traffic Counts Hourly Permanent",
            "format": "GZIP",  # not handled
            "datastore_active": False,
            "url": "https://example.com/hourly.gz",
        },
    ]
    responses.add(
        responses.GET,
        f"{traffic.API_ROOT}/package_show",
        json=_package_show_payload(resources),
        status=200,
    )
    responses.add(
        responses.GET,
        f"{traffic.API_ROOT}/datastore_search",
        json=_datastore_payload([], total=0),
        status=200,
        match=[responses.matchers.query_param_matcher(
            {"resource_id": "stations-id", "limit": "10000", "offset": "0"}
        )],
    )

    with pytest.raises(RuntimeError, match="don't know how to read"):
        traffic.fetch("2024-01-01", "2024-12-31", tmp_path)
