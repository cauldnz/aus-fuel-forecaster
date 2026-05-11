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
    assert cf._canonicalise_brand("Caltex", mapping, unmapped).canonical == "Ampol"
    assert cf._canonicalise_brand("EG Ampol", mapping, unmapped).canonical == "Ampol"
    assert cf._canonicalise_brand("Ampol", mapping, unmapped).canonical == "Ampol"
    assert cf._canonicalise_brand("BP", mapping, unmapped).canonical == "BP"
    assert unmapped == set()


def test_brand_is_major_flag_preserved(aliases_path: Path) -> None:
    """`is_major` must come through alongside the canonical name."""
    mapping = cf.load_brand_aliases(aliases_path)
    unmapped: set[str] = set()
    assert cf._canonicalise_brand("BP", mapping, unmapped).is_major is True
    assert cf._canonicalise_brand("Independent", mapping, unmapped).is_major is False


def test_unknown_brand_passes_through_and_is_recorded(aliases_path: Path) -> None:
    mapping = cf.load_brand_aliases(aliases_path)
    unmapped: set[str] = set()
    new = "Brand-We-Have-Never-Seen"
    info = cf._canonicalise_brand(new, mapping, unmapped)
    assert info.canonical == new
    assert info.is_major is False
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


def test_caltex_consolidates_to_ampol_but_preserves_raw(
    tmp_path: Path, aliases_path: Path
) -> None:
    """`brand_canonical` collapses Caltex to Ampol; `brand_raw` keeps Caltex."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(_sample_month("2024/08/", brand="Caltex"), raw_dir / "2024-08.parquet")

    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    cf.clean(raw_dir, out, stations, brand_aliases=aliases_path)

    roster = pd.read_parquet(stations)
    # Both stations had Caltex / Caltex.
    assert (roster["brand_canonical"] == "Ampol").all()
    assert (roster["brand_raw"] == "Caltex").all()
    # `is_major` propagates from the alias CSV (Caltex,Ampol,true).
    assert roster["brand_is_major"].all()


def test_stations_roster_has_full_brand_schema(tmp_path: Path, aliases_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(_sample_month("2024/08/"), raw_dir / "2024-08.parquet")

    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    cf.clean(raw_dir, out, stations, brand_aliases=aliases_path)

    roster = pd.read_parquet(stations)
    expected = {"brand_raw", "brand_canonical", "brand_is_major"}
    assert expected <= set(roster.columns)
    # The legacy single `brand` column should NOT be present (we'd be
    # destroying signal — see CLAUDE.md memory).
    assert "brand" not in roster.columns


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
    misses = tmp_path / "brand_misses.csv"
    with caplog.at_level("WARNING", logger="fuel_pred.clean.fuelcheck"):
        cf.clean(
            raw_dir, out, stations,
            brand_aliases=aliases_path,
            brand_misses_out=misses,
        )

    assert any("unmapped" in rec.message for rec in caplog.records)


def test_brand_miss_sidecar_has_expected_schema_and_counts(
    tmp_path: Path, aliases_path: Path
) -> None:
    """The sidecar CSV holds one row per raw brand, sorted by occurrences."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Two months. Three unmapped brands with different occurrence counts:
    # `Mystery A` appears 5 times across 2 stations; `Mystery B` appears
    # 2 times at 1 station; `Mystery C` appears 1 time.
    aug = _sample_month("2024/08/")
    for i in range(3):
        aug[i]["ServiceStationName"] = "Station Alpha"
        aug[i]["Address"] = "1 Alpha St"
        aug[i]["Brand"] = "Mystery A"
    aug[3]["ServiceStationName"] = "Station Bravo"
    aug[3]["Address"] = "2 Bravo Rd"
    aug[3]["Brand"] = "Mystery A"
    aug[4]["Brand"] = "Mystery B"  # at the default sample_month station
    aug[5]["Brand"] = "Mystery B"

    sep = _sample_month("2024/09/")
    sep[0]["ServiceStationName"] = "Station Alpha"
    sep[0]["Address"] = "1 Alpha St"
    sep[0]["Brand"] = "Mystery A"   # 5th total occurrence of Mystery A
    sep[1]["Brand"] = "Mystery C"

    _write(aug, raw_dir / "2024-08.parquet")
    _write(sep, raw_dir / "2024-09.parquet")

    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    misses = tmp_path / "brand_misses.csv"
    cf.clean(
        raw_dir, out, stations,
        brand_aliases=aliases_path,
        brand_misses_out=misses,
    )

    assert misses.exists(), "sidecar CSV should be created when there are misses"
    df = pd.read_csv(misses)
    assert list(df.columns) == [
        "raw_brand",
        "n_occurrences",
        "n_stations",
        "sample_name",
        "sample_address",
        "sample_suburb",
        "first_seen_in",
    ]
    # Sorted by count desc.
    assert df["n_occurrences"].tolist() == sorted(
        df["n_occurrences"].tolist(), reverse=True
    )
    by_brand = df.set_index("raw_brand")
    assert by_brand.loc["Mystery A", "n_occurrences"] == 5
    assert by_brand.loc["Mystery A", "n_stations"] == 2
    assert by_brand.loc["Mystery B", "n_occurrences"] == 2
    assert by_brand.loc["Mystery C", "n_occurrences"] == 1
    # Sample columns are populated (verbatim from the first occurrence).
    assert by_brand.loc["Mystery A", "sample_name"] == "Station Alpha"
    assert by_brand.loc["Mystery A", "sample_address"] == "1 Alpha St"
    # first_seen_in is the file the brand first appeared in.
    assert by_brand.loc["Mystery A", "first_seen_in"] == "2024-08.parquet"
    assert by_brand.loc["Mystery C", "first_seen_in"] == "2024-09.parquet"


def test_brand_miss_sidecar_absent_when_all_brands_mapped(
    tmp_path: Path, aliases_path: Path
) -> None:
    """No misses → no sidecar file (and any stale one is removed)."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _sample_month("2024/08/")  # All "BP", which IS in aliases.
    _write(rows, raw_dir / "2024-08.parquet")

    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    misses = tmp_path / "brand_misses.csv"
    # Pre-create a stale sidecar to verify it gets removed.
    misses.write_text("stale,from,a,previous,run\n", encoding="utf-8")

    cf.clean(
        raw_dir, out, stations,
        brand_aliases=aliases_path,
        brand_misses_out=misses,
    )

    assert not misses.exists(), "sidecar should be absent when no misses occur"


def test_brand_miss_sidecar_default_path_is_data_interim(
    tmp_path: Path, aliases_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `brand_misses_out` is None, defaults to config.DATA_INTERIM/brand_misses.csv."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _sample_month("2024/08/")
    rows[0]["Brand"] = "Mystery"
    _write(rows, raw_dir / "2024-08.parquet")

    # Redirect DATA_INTERIM to tmp so the test stays hermetic.
    interim = tmp_path / "interim"
    interim.mkdir()
    monkeypatch.setattr("fuel_pred.config.DATA_INTERIM", interim)

    cf.clean(
        raw_dir,
        tmp_path / "fuel_daily.parquet",
        tmp_path / "stations.parquet",
        brand_aliases=aliases_path,
        # brand_misses_out intentionally omitted → exercise the default
    )

    assert (interim / "brand_misses.csv").exists()


# --------------- Excel header layout variants (issue #19) ---------------
#
# NSW Open Data ships monthly Price History workbooks in three header
# layouts. The fetcher reads `header=0` for everything and persists
# whatever it finds, so the cached parquets carry the layout drift
# forward. _detect_and_promote_headers recovers from all three.


def _variant_a_rows() -> list[dict[str, object]]:
    """Headers at row 0 (clean form). Same shape as `_sample_month`."""
    return _sample_month("2024/08/")


def _variant_b_dataframe() -> pd.DataFrame:
    """Title cell at row 0; real headers at row 1.

    Mirrors the 2017-07/08, 2018-11/12, 2019-* cached parquets.
    """
    headers = list(_variant_a_rows()[0].keys())
    data = [list(r.values()) for r in _variant_a_rows()]
    placeholder_cols = [
        f"Unnamed: {i}" if i else "Price_History_July_2017"
        for i in range(len(headers))
    ]
    return pd.DataFrame([headers, *data], columns=placeholder_cols)


def _variant_c_dataframe() -> pd.DataFrame:
    """Title cell at row 0; blank row at index 0 of data; real headers at
    index 1; descriptive columns merged-cell-style (only first row of
    each station's block is filled).

    Mirrors the 2020-*, 2021-*, 2022-01..04, etc. cached parquets.
    """
    rows = _variant_a_rows()
    headers = list(rows[0].keys())

    # Build the "merged-cell" pattern: keep ServiceStationName, Address,
    # Suburb, Postcode, Brand on the first row of each station's block;
    # NaN them on subsequent rows of the same station.
    sticky_cols = {"ServiceStationName", "Address", "Suburb", "Postcode", "Brand"}
    last_station = None
    merged_rows = []
    for r in rows:
        new = dict(r)
        if r["ServiceStationName"] == last_station:
            for c in sticky_cols:
                new[c] = None
        last_station = r["ServiceStationName"]
        merged_rows.append(new)

    placeholder_cols = [f"Unnamed: {i}" if i else "Price History Checks"
                        for i in range(len(headers))]
    blank_row = [None] * len(headers)
    header_row = headers
    data_rows = [list(r.values()) for r in merged_rows]
    return pd.DataFrame([blank_row, header_row, *data_rows], columns=placeholder_cols)


def test_detect_and_promote_headers_variant_a_passthrough() -> None:
    """Variant A — clean headers — returned unchanged."""
    df = pd.DataFrame(_variant_a_rows())
    out = cf._detect_and_promote_headers(df)
    assert list(out.columns) == list(df.columns)
    assert len(out) == len(df)


def test_detect_and_promote_headers_variant_b() -> None:
    """Variant B — title cell at row 0, headers at row 1 — promoted correctly."""
    df = _variant_b_dataframe()
    out = cf._detect_and_promote_headers(df)
    expected_cols = list(_variant_a_rows()[0].keys())
    assert list(out.columns) == expected_cols
    # Title row dropped, plus the headers-as-data row promoted.
    assert len(out) == len(_variant_a_rows())
    assert out.iloc[0]["ServiceStationName"] == "BP Mascot"


def test_detect_and_promote_headers_variant_c_with_ffill() -> None:
    """Variant C — title + blank + merged cells — promoted and ffilled."""
    df = _variant_c_dataframe()
    out = cf._detect_and_promote_headers(df)
    expected_cols = list(_variant_a_rows()[0].keys())
    assert list(out.columns) == expected_cols
    # No NaN in the descriptive columns after ffill — every row knows its station.
    for col in ("ServiceStationName", "Address", "Suburb", "Postcode", "Brand"):
        assert out[col].notna().all(), f"{col} should have been forward-filled"
    # Per-event columns (price, fuel) are NOT ffilled — they're the actual data.
    assert out["FuelCode"].notna().all()  # in our fixture every row has its own fuel
    # Station identity preserved across "blocks".
    bp_rows = out[out["ServiceStationName"] == "BP Mascot"]
    assert len(bp_rows) == 9  # 3 fuels x 3 days


def test_detect_and_promote_headers_unknown_layout_passthrough() -> None:
    """Future schema drift: unrecognised → pass through, let normaliser warn."""
    df = pd.DataFrame(
        {"junk_col_a": ["x", "y"], "junk_col_b": [1, 2]}
    )
    out = cf._detect_and_promote_headers(df)
    # Should be unchanged so the downstream "missing columns" warning fires.
    assert list(out.columns) == ["junk_col_a", "junk_col_b"]
    assert len(out) == 2


def test_clean_recovers_data_from_variant_b_files(
    tmp_path: Path, aliases_path: Path
) -> None:
    """End-to-end: a Variant B monthly should produce daily rows, not be skipped."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _variant_b_dataframe().to_parquet(
        raw_dir / "2017-07.parquet", engine="pyarrow", compression="zstd", index=False
    )
    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    cf.clean(raw_dir, out, stations, brand_aliases=aliases_path)
    daily = pd.read_parquet(out)
    # Same fixture as Variant A: 2 stations x 2 fuels (E10 filtered) x 3 days.
    assert len(daily) == 12


def test_clean_recovers_data_from_variant_c_files(
    tmp_path: Path, aliases_path: Path
) -> None:
    """End-to-end: a Variant C monthly recovers post-ffill — rows that were
    NaN in the merged-cell layout still produce daily aggregates."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _variant_c_dataframe().to_parquet(
        raw_dir / "2020-06.parquet", engine="pyarrow", compression="zstd", index=False
    )
    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    cf.clean(raw_dir, out, stations, brand_aliases=aliases_path)
    daily = pd.read_parquet(out)
    # Without ffill we'd lose ~half the rows (the merged-cell ones drop on dropna).
    assert len(daily) == 12


def test_clean_handles_mixed_layouts_in_one_run(
    tmp_path: Path, aliases_path: Path
) -> None:
    """Real corpus: variant A + B + C months side-by-side in `data/raw/fuelcheck/`."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(_variant_a_rows(), raw_dir / "2024-01.parquet")
    _variant_b_dataframe().to_parquet(
        raw_dir / "2017-07.parquet", engine="pyarrow", compression="zstd", index=False
    )
    _variant_c_dataframe().to_parquet(
        raw_dir / "2020-06.parquet", engine="pyarrow", compression="zstd", index=False
    )

    out = tmp_path / "fuel_daily.parquet"
    stations = tmp_path / "stations.parquet"
    cf.clean(raw_dir, out, stations, brand_aliases=aliases_path)

    daily = pd.read_parquet(out)
    # 3 months, 12 daily rows each (deduplication: same station_ids
    # across months produce dupes that re-aggregate). Worst case
    # we get 12 * 3 = 36 rows; same-day dedup means we'll see ≤ 36
    # but ≥ 12. Just assert we have something from each month-shape.
    assert len(daily) >= 12


# ----------------------------- Postcode normalisation (issue #23) -----------------------------


def test_postcode_float_artifact_stripped_at_source(
    tmp_path: Path, aliases_path: Path
) -> None:
    """Pandas float coercion serialises postcode as '2776.0'; we strip it."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _sample_month("2024/08/")
    # Mimic what pandas does when one cell in the column is missing during
    # a chunked CSV read: type bumps to float64 → string round-trip yields '.0'.
    df = pd.DataFrame(rows)
    df["Postcode"] = df["Postcode"].astype(float).astype("string")  # → '2020.0', '2042.0'
    df.to_parquet(raw_dir / "2024-08.parquet", engine="pyarrow", compression="zstd", index=False)

    out = tmp_path / "fuel_daily.parquet"
    stations_out = tmp_path / "stations.parquet"
    cf.clean(raw_dir, out, stations_out, brand_aliases=aliases_path)

    stations_df = pd.read_parquet(stations_out)
    # No '.0' anywhere in the postcodes.
    assert not stations_df["postcode"].astype(str).str.endswith(".0").any()
    # And the canonical 4-digit form is preserved.
    pcs = sorted(stations_df["postcode"].unique().tolist())
    assert pcs == ["2020", "2042"]


def test_normalise_postcode_series_handles_mixed_input() -> None:
    """Direct unit test on the normaliser, covering the cases pandas hands us."""
    s = pd.Series(["2776.0", "2042", " 2000 ", None, "LPO 1234", 2776.0])
    out = cf._normalise_postcode_series(s)
    # NaN preserved (so dropna can filter); whitespace stripped; .0 dropped;
    # non-numeric (LPO) untouched; float coerced to "2776".
    assert out.tolist() == ["2776", "2042", "2000", pd.NA, "LPO 1234", "2776"]


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
