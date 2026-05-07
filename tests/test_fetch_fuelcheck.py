"""Hermetic tests for fetch.fuelcheck.

CKAN package_show + per-resource CSV downloads are mocked with `responses`.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import responses

from fuel_pred.fetch import fuelcheck


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _resource(
    name: str,
    *,
    url: str = "https://example.com/fuel.csv",
    fmt: str = "CSV",
    rid: str = "r-id",
) -> dict[str, Any]:
    return {"id": rid, "name": name, "url": url, "format": fmt}


def _package_payload(resources: list[dict[str, Any]]) -> dict[str, Any]:
    return {"success": True, "result": {"resources": resources}}


@pytest.fixture
def sample_csv() -> bytes:
    return _csv_bytes(
        [
            {
                "ServiceStationName": "Caltex Mascot",
                "Address": "1 Botany Rd",
                "Suburb": "Mascot",
                "Postcode": "2020",
                "Brand": "Caltex",
                "FuelCode": "U91",
                "PriceUpdatedDate": "2024/08/01 12:34:56",
                "Price": "189.9",
            },
            {
                "ServiceStationName": "BP Newtown",
                "Address": "100 King St",
                "Suburb": "Newtown",
                "Postcode": "2042",
                "Brand": "BP",
                "FuelCode": "DL",
                "PriceUpdatedDate": "2024/08/02 09:00:00",
                "Price": "210.5",
            },
        ]
    )


# ----------------------------- extract_year_month -----------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Service Station Price History - August 2024", (2024, 8)),
        ("Price History 2024-08", (2024, 8)),
        ("price_history_2024_08.csv", (2024, 8)),
        ("PriceHistory_AUG-2024.csv", (2024, 8)),
        ("price-history-aug2024.csv", (2024, 8)),
        ("Price History September 2016", (2016, 9)),
        ("Service Station and Brand Reference 2024", None),  # no clear month
    ],
)
def test_extract_year_month(text: str, expected: tuple[int, int] | None) -> None:
    assert fuelcheck.extract_year_month(text) == expected


def test_resource_filter_excludes_brand_reference() -> None:
    assert not fuelcheck._is_price_history_resource(
        _resource("Service Station and Brand Reference - August 2024")
    )
    assert fuelcheck._is_price_history_resource(
        _resource("Service Station Price History - August 2024")
    )


# ----------------------------- fetch -----------------------------


@responses.activate
def test_fetches_only_months_in_range(tmp_path: Path, sample_csv: bytes) -> None:
    """Only months whose data overlaps [start, end] should be downloaded."""
    resources = [
        _resource(
            "Service Station Price History - July 2024",
            url="https://example.com/jul.csv",
            rid="r-jul",
        ),
        _resource(
            "Service Station Price History - August 2024",
            url="https://example.com/aug.csv",
            rid="r-aug",
        ),
        _resource(
            "Service Station Price History - September 2024",
            url="https://example.com/sep.csv",
            rid="r-sep",
        ),
        _resource(
            "Service Station and Brand Reference - August 2024",
            url="https://example.com/brand.csv",
            rid="r-brand",
        ),
    ]
    responses.add(
        responses.GET,
        f"{fuelcheck.API_ROOT}/package_show",
        json=_package_payload(resources),
        status=200,
    )
    responses.add(responses.GET, "https://example.com/aug.csv", body=sample_csv, status=200)

    fuelcheck.fetch("2024-08-01", "2024-08-31", tmp_path)

    files = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert files == ["2024-08.parquet"]
    df = pd.read_parquet(tmp_path / "2024-08.parquet")
    assert len(df) == 2
    # Schema not enforced — verify the raw columns were preserved.
    assert "ServiceStationName" in df.columns


@responses.activate
def test_skips_cached_files_unless_force(tmp_path: Path, sample_csv: bytes) -> None:
    """An existing parquet for a month should not be re-downloaded by default."""
    (tmp_path / "2024-08.parquet").write_bytes(b"cached-placeholder")

    resources = [
        _resource(
            "Service Station Price History - August 2024",
            url="https://example.com/aug.csv",
            rid="r-aug",
        ),
    ]
    responses.add(
        responses.GET,
        f"{fuelcheck.API_ROOT}/package_show",
        json=_package_payload(resources),
        status=200,
    )
    # No mock for the CSV — if fetch tries to download it, the test fails.

    fuelcheck.fetch("2024-08-01", "2024-08-31", tmp_path)
    assert (tmp_path / "2024-08.parquet").read_bytes() == b"cached-placeholder"

    # With force=True, it must re-download.
    responses.add(responses.GET, "https://example.com/aug.csv", body=sample_csv, status=200)
    fuelcheck.fetch("2024-08-01", "2024-08-31", tmp_path, force=True)
    df = pd.read_parquet(tmp_path / "2024-08.parquet")
    assert len(df) == 2


@responses.activate
def test_handles_schema_drift_gracefully(tmp_path: Path) -> None:
    """A month with an unexpected column set must still be written verbatim."""
    drifted = _csv_bytes(
        [
            {
                "service_station_name": "Caltex Mascot",  # snake_case rename
                "address": "1 Botany Rd",
                "suburb": "Mascot",
                "postcode": "2020",
                "brand": "Caltex",
                "fuel_code": "U91",
                "price_updated_date": "2024-09-01T12:34:56Z",  # ISO format
                "price": "189.9",
                "extra_column": "new!",  # extra column
            }
        ]
    )
    resources = [
        _resource(
            "Service Station Price History - September 2024",
            url="https://example.com/sep.csv",
            rid="r-sep",
        ),
    ]
    responses.add(
        responses.GET,
        f"{fuelcheck.API_ROOT}/package_show",
        json=_package_payload(resources),
        status=200,
    )
    responses.add(responses.GET, "https://example.com/sep.csv", body=drifted, status=200)

    fuelcheck.fetch("2024-09-01", "2024-09-30", tmp_path)

    df = pd.read_parquet(tmp_path / "2024-09.parquet")
    assert "extra_column" in df.columns
    assert "service_station_name" in df.columns
    assert df["extra_column"].iloc[0] == "new!"


@responses.activate
def test_no_resources_in_range_logs_and_returns(tmp_path: Path) -> None:
    resources = [
        _resource(
            "Service Station Price History - January 2020",
            url="https://example.com/jan.csv",
            rid="r-jan",
        ),
    ]
    responses.add(
        responses.GET,
        f"{fuelcheck.API_ROOT}/package_show",
        json=_package_payload(resources),
        status=200,
    )

    fuelcheck.fetch("2024-01-01", "2024-12-31", tmp_path)
    assert list(tmp_path.glob("*.parquet")) == []


@responses.activate
def test_resource_with_unparseable_name_is_skipped(tmp_path: Path) -> None:
    resources = [
        _resource(
            "Price History (random-blob)",  # no month
            url="https://example.com/x.csv",
            rid="r-x",
        ),
        _resource(
            "Service Station Price History - August 2024",
            url="https://example.com/aug.csv",
            rid="r-aug",
        ),
    ]
    responses.add(
        responses.GET,
        f"{fuelcheck.API_ROOT}/package_show",
        json=_package_payload(resources),
        status=200,
    )
    responses.add(
        responses.GET,
        "https://example.com/aug.csv",
        body=_csv_bytes([{"a": "1"}]),
        status=200,
    )

    fuelcheck.fetch("2024-01-01", "2024-12-31", tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == ["2024-08.parquet"]


@responses.activate
def test_non_tabular_resource_filtered_out(tmp_path: Path) -> None:
    """Unsupported formats (zip, pdf) are skipped even if the name matches."""
    resources = [
        _resource(
            "Service Station Price History - August 2024",
            url="https://example.com/aug.zip",
            fmt="ZIP",
            rid="r-aug",
        ),
    ]
    responses.add(
        responses.GET,
        f"{fuelcheck.API_ROOT}/package_show",
        json=_package_payload(resources),
        status=200,
    )

    fuelcheck.fetch("2024-08-01", "2024-08-31", tmp_path)
    assert list(tmp_path.glob("*.parquet")) == []


@responses.activate
def test_xlsx_resource_is_fetched(tmp_path: Path) -> None:
    """Most NSW FuelCheck monthly archives are XLSX, not CSV."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(["ServiceStationName", "Brand", "FuelCode", "Price"])
    ws.append(["Caltex Mascot", "Caltex", "U91", "189.9"])
    ws.append(["BP Newtown", "BP", "DL", "210.5"])
    buf = io.BytesIO()
    wb.save(buf)
    xlsx_bytes = buf.getvalue()

    resources = [
        _resource(
            "FuelCheck Price History August 2024",
            url="https://example.com/aug.xlsx",
            fmt="XLSX",
            rid="r-aug",
        ),
    ]
    responses.add(
        responses.GET,
        f"{fuelcheck.API_ROOT}/package_show",
        json=_package_payload(resources),
        status=200,
    )
    responses.add(responses.GET, "https://example.com/aug.xlsx", body=xlsx_bytes, status=200)

    fuelcheck.fetch("2024-08-01", "2024-08-31", tmp_path)

    df = pd.read_parquet(tmp_path / "2024-08.parquet")
    assert len(df) == 2
    assert list(df.columns) == ["ServiceStationName", "Brand", "FuelCode", "Price"]


@responses.activate
def test_empty_format_with_csv_url_is_fetched(tmp_path: Path, sample_csv: bytes) -> None:
    """Some real resources have an empty format string; trust the URL extension."""
    resources = [
        _resource(
            "FuelCheck Price History July 2024",
            url="https://example.com/jul.csv",
            fmt="",
            rid="r-jul",
        ),
    ]
    responses.add(
        responses.GET,
        f"{fuelcheck.API_ROOT}/package_show",
        json=_package_payload(resources),
        status=200,
    )
    responses.add(responses.GET, "https://example.com/jul.csv", body=sample_csv, status=200)

    fuelcheck.fetch("2024-07-01", "2024-07-31", tmp_path)
    assert (tmp_path / "2024-07.parquet").exists()
