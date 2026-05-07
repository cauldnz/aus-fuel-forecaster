"""Hermetic tests for clean.fuelcheck — synthetic monthly Parquets only."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fuel_pred.clean import fuelcheck as cf

# ----------------------------- helpers -----------------------------


def _make_monthly(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _write(rows: list[dict[str, object]], path: Path) -> Path:
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


@pytest.fixture
def aliases_path(tmp_path: Path) -> Path:
    p = tmp_path / "brand_aliases.csv"
    p.write_text(
        "raw_brand,canonical_brand,is_major\n"
        "BP,BP,true\n"
        "Caltex,Ampol,true\n"
        "EG Ampol,Ampol,true\n"
        "Ampol,Ampol,true\n"
        "Independent,Independent,false\n",
        encoding="utf-8",
    )
    return p


# ----------------------------- pure helpers -----------------------------


def test_station_id_is_stable_across_runs() -> None:
    """Same (name, address, suburb, postcode) → same station_id."""
    a = cf._hash_station("BP Mascot", "1 Botany Rd", "Mascot", "2020")
    b = cf._hash_station("BP Mascot", "1 Botany Rd", "Mascot", "2020")
    assert a == b
    # And it's deterministic across cosmetic whitespace / case.
    c = cf._hash_station("  bp mascot ", "1 botany rd", "MASCOT", "2020")
    assert c == a


def test_station_id_differs_for_different_addresses() -> None:
    a = cf._hash_station("BP Mascot", "1 Botany Rd", "Mascot", "2020")
    b = cf._hash_station("BP Mascot", "2 Botany Rd", "Mascot", "2020")
    assert a != b


def test_brand_aliasing_canonicalises_known_variants(aliases_path: Path) -> None:
    mapping = cf.load_brand_aliases(aliases_path)
    unmapped: set[str] = set()
    assert cf._canonicalise_brand("Caltex", mapping, unmapped) == "Ampol"
    assert cf._canonicalise_brand("EG Ampol", mapping, unmapped) == "Ampol"
    assert cf._canonicalise_brand("Ampol", mapping, unmapped) == "Ampol"
    assert cf._canonicalise_brand("BP", mapping, unmapped) == "BP"
    assert unmapped == set()


def test_unknown_brand_passes_through_and_is_recorded(aliases_path: Path) -> None:
    mapping = cf.load_brand_aliases(aliases_path)
    unmapped: set[str] = set()
    new = "Brand-We-Have-Never-Seen"
    assert cf._canonicalise_brand(new, mapping, unmapped) == new
    assert new in unmapped


def test_normalise_columns_handles_snake_case_drift() -> None:
    df = pd.DataFrame(
        {
            "service_station_name": ["X"],
            "address": ["1 X St"],
            "suburb": ["Foo"],
            "postcode": ["2000"],
            "brand": ["BP"],
            "fuel_code": ["U91"],
            "price_updated_date": ["2024-08-01T00:00:00Z"],
            "price": ["189.9"],
            "extra_column_to_drop": ["junk"],
        }
    )
    out = cf._normalise_columns(df)
    assert list(out.columns) == list(cf.REQUIRED_COLUMNS)
    assert out["name"].iloc[0] == "X"


def test_parse_price_date_handles_multiple_formats() -> None:
    parsed = cf._parse_price_date(
        pd.Series(
            [
                "2024/08/01 12:34:56",
                "2024-09-01T12:34:56Z",
                "2024-10-01",
                "garbage",
            ]
        )
    )
    import datetime as dt

    assert parsed.iloc[0] == dt.date(2024, 8, 1)
    assert parsed.iloc[1] == dt.date(2024, 9, 1)
    assert parsed.iloc[2] == dt.date(2024, 10, 1)
    assert pd.isna(parsed.iloc[3])


# ----------------------------- end-to-end clean() -----------------------------


def _sample_month(start: str, brand: str = "BP") -> list[dict[str, object]]:
    """Two stations x U91 + DL x 3 days each."""
    rows: list[dict[str, object]] = []
    for station_idx, name in enumerate(["BP Mascot", "Caltex Newtown"], start=1):
        for fuel in ("U91", "DL", "E10"):  # E10 should be filtered out
            for day in range(1, 4):
                rows.append(
                    {
                        "ServiceStationName": name,
                        "Address": f"{station_idx} Main St",
                        "Suburb": "Mascot" if station_idx == 1 else "Newtown",
                        "Postcode": "2020" if station_idx == 1 else "2042",
                        "Brand": brand if station_idx == 1 else "Caltex",
                        "FuelCode": fuel,
                        "PriceUpdatedDate": f"{start[:8]}{day:02d} 12:00:00",
                        "Price": "189.9",
                    }
                )
    return rows


def test_daily_aggregation_produces_one_row_per_station_fuel_day(
    tmp_path: Path, aliases_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(_sample_month("2024/08/"), raw_dir / "2024-08.parquet")

    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    cf.clean(raw_dir, out, stations, brand_aliases=aliases_path)

    daily = pd.read_parquet(out)
    # 2 stations x 2 fuels (E10 filtered) x 3 days = 12 rows.
    assert len(daily) == 12
    assert set(daily["fuel_code"]) == {"U91", "DL"}
    # Per-key uniqueness.
    assert not daily.duplicated(["station_id", "fuel_code", "date"]).any()
    # Schema per spec §6.2.
    expected = {"station_id", "fuel_code", "date", "price_mean", "price_min", "price_max", "n_obs"}
    assert expected <= set(daily.columns)


def test_clean_concatenates_multiple_months(tmp_path: Path, aliases_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(_sample_month("2024/08/"), raw_dir / "2024-08.parquet")
    _write(_sample_month("2024/09/"), raw_dir / "2024-09.parquet")

    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    cf.clean(raw_dir, out, stations, brand_aliases=aliases_path)

    daily = pd.read_parquet(out)
    # 2 months x 2 stations x 2 fuels x 3 days = 24 rows.
    assert len(daily) == 24

    roster = pd.read_parquet(stations)
    # Two stations.
    assert len(roster) == 2
    # first_seen / last_seen span both months.
    import datetime as dt

    assert roster["first_seen"].min() == dt.date(2024, 8, 1)
    assert roster["last_seen"].max() == dt.date(2024, 9, 3)


def test_caltex_consolidates_to_ampol(tmp_path: Path, aliases_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(_sample_month("2024/08/", brand="Caltex"), raw_dir / "2024-08.parquet")

    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    cf.clean(raw_dir, out, stations, brand_aliases=aliases_path)

    roster = pd.read_parquet(stations)
    # Both stations had Caltex / Caltex; the aliases collapse both to Ampol.
    assert (roster["brand"] == "Ampol").all()


def test_warn_on_unmapped_brand(
    tmp_path: Path, aliases_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _sample_month("2024/08/")
    # Inject one row whose brand isn't in the aliases CSV.
    rows[0]["Brand"] = "Brand-We-Have-Never-Seen"
    _write(rows, raw_dir / "2024-08.parquet")

    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    with caplog.at_level("WARNING", logger="fuel_pred.clean.fuelcheck"):
        cf.clean(raw_dir, out, stations, brand_aliases=aliases_path)

    assert any("unmapped" in rec.message for rec in caplog.records)


def test_empty_input_dir_raises(tmp_path: Path, aliases_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with pytest.raises(RuntimeError, match="no monthly parquets"):
        cf.clean(
            raw_dir,
            tmp_path / "out.parquet",
            tmp_path / "stations.parquet",
            brand_aliases=aliases_path,
        )
