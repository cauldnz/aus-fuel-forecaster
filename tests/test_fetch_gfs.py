"""Hermetic tests for fetch.gfs — URL routing, idx parsing, byte-range fetch, GRIB parse.

Real-network smoke is in `tools/` (separate, opt-in). These tests use:
  - small synthetic .idx text for parser tests
  - the saved real-GRIB fixtures under `tests/fixtures/gfs/` for parse tests
    (each <100 KB; one-time fetch from the live S3 bucket).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import responses
import xarray as xr

from fuel_pred.fetch import gfs

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gfs"


# ----------------------------- _select_resolution_for_date -----------------------------


def test_select_resolution_gefs_1deg_window() -> None:
    """2017-01-01 .. 2020-09-22 routes to GEFS 1°."""
    assert gfs._select_resolution_for_date(dt.date(2017, 1, 1)) == "gefs_1"
    assert gfs._select_resolution_for_date(dt.date(2018, 6, 1)) == "gefs_1"
    assert gfs._select_resolution_for_date(dt.date(2020, 9, 22)) == "gefs_1"


def test_select_resolution_gefs_05deg_bridge() -> None:
    """2020-09-23 .. 2021-03-31 routes to GEFS 0.5° (v12 ensemble mean)."""
    assert gfs._select_resolution_for_date(dt.date(2020, 9, 23)) == "gefs_0p5"
    assert gfs._select_resolution_for_date(dt.date(2021, 1, 1)) == "gefs_0p5"
    assert gfs._select_resolution_for_date(dt.date(2021, 3, 31)) == "gefs_0p5"


def test_select_resolution_gfs_025deg_present() -> None:
    """2021-04-01 onward routes to GFS 0.25°."""
    assert gfs._select_resolution_for_date(dt.date(2021, 4, 1)) == "gfs_0p25"
    assert gfs._select_resolution_for_date(dt.date(2023, 6, 1)) == "gfs_0p25"
    assert gfs._select_resolution_for_date(dt.date(2026, 5, 1)) == "gfs_0p25"


def test_select_resolution_rejects_pre_2017() -> None:
    with pytest.raises(ValueError, match="before GFS/GEFS coverage start"):
        gfs._select_resolution_for_date(dt.date(2016, 12, 31))


# ----------------------------- _build_url (boundary cases) -----------------------------


def test_build_url_gefs_layout1_pre_2019() -> None:
    """2018-12-31: pre-2019 GEFS 1° layout — top-level, 3-digit lead."""
    url = gfs._build_url(dt.date(2018, 12, 31), "00", 24, "gefs_1")
    assert url == (
        "https://noaa-gefs-pds.s3.amazonaws.com/"
        "gefs.20181231/00/gec00.t00z.pgrb2af024"
    )


def test_build_url_gefs_layout2_2019_onward() -> None:
    """2019-01-01: GEFS 1° layout-2 — pgrb2a subdir, 2-digit lead."""
    url = gfs._build_url(dt.date(2019, 1, 1), "00", 24, "gefs_1")
    assert url == (
        "https://noaa-gefs-pds.s3.amazonaws.com/"
        "gefs.20190101/00/pgrb2a/gec00.t00z.pgrb2af24"
    )


def test_build_url_gefs_layout1_to_layout2_transition() -> None:
    """The layout switches sharply at 2019-01-01 — pre uses 3-digit, post 2-digit."""
    pre = gfs._build_url(dt.date(2018, 12, 31), "00", 6, "gefs_1")
    post = gfs._build_url(dt.date(2019, 1, 1), "00", 6, "gefs_1")
    assert "pgrb2af006" in pre  # 3-digit, top level
    # Layout-2 uses 2-digit zero-padded lead (`f06`, not `f6`) in pgrb2a/ subdir.
    # Confirmed empirically: `pgrb2af6.idx` -> 404; `pgrb2af06.idx` -> 200.
    assert "pgrb2a/gec00.t00z.pgrb2af06" in post


def test_build_url_gefs_v12_05deg_2020_onward() -> None:
    """2020-09-23 (and beyond): GEFSv12 0.5° — pgrb2ap5 subdir."""
    url = gfs._build_url(dt.date(2020, 9, 23), "00", 24, "gefs_0p5")
    assert url == (
        "https://noaa-gefs-pds.s3.amazonaws.com/"
        "gefs.20200923/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f024"
    )


def test_build_url_gfs_025deg_2021_onward() -> None:
    """2021-04-01 onward: GFS 0.25° at noaa-gfs-bdp-pds."""
    url = gfs._build_url(dt.date(2021, 4, 1), "00", 24, "gfs_0p25")
    assert url == (
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        "gfs.20210401/00/atmos/gfs.t00z.pgrb2.0p25.f024"
    )


def test_build_url_gfs_supports_long_leads() -> None:
    """Day-7 horizon (lead 168) renders as 3-digit padded."""
    url = gfs._build_url(dt.date(2023, 6, 1), "00", 168, "gfs_0p25")
    assert url.endswith("pgrb2.0p25.f168")


def test_build_url_rejects_bad_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        gfs._build_url(dt.date(2023, 6, 1), "01", 24, "gfs_0p25")


def test_build_url_rejects_negative_lead() -> None:
    with pytest.raises(ValueError, match="lead_h"):
        gfs._build_url(dt.date(2023, 6, 1), "00", -1, "gfs_0p25")


# ----------------------------- _fetch_idx parsing -----------------------------


_FIXTURE_IDX = """1:0:d=2018060100:PRMSL:mean sea level:24 hour fcst:ENS=low-res ctl
2:100:d=2018060100:HGT:1000 mb:24 hour fcst:ENS=low-res ctl
3:200:d=2018060100:TMP:2 m above ground:24 hour fcst:ENS=low-res ctl
4:300:d=2018060100:TMAX:2 m above ground:18-24 hour max fcst:ENS=low-res ctl
5:400:d=2018060100:TMIN:2 m above ground:18-24 hour min fcst:ENS=low-res ctl
6:500:d=2018060100:UGRD:10 m above ground:24 hour fcst:ENS=low-res ctl
7:600:d=2018060100:VGRD:10 m above ground:24 hour fcst:ENS=low-res ctl
8:700:d=2018060100:APCP:surface:18-24 hour acc fcst:ENS=low-res ctl
"""


@responses.activate
def test_fetch_idx_parses_byte_ranges() -> None:
    """Parser produces (start, end) byte ranges keyed by `VAR:level`."""
    url = "https://example.s3.amazonaws.com/fake.idx"
    responses.add(responses.GET, url, body=_FIXTURE_IDX, status=200)

    idx = gfs._fetch_idx(url)

    assert idx["TMAX:2 m above ground"] == (300, 400)
    assert idx["TMIN:2 m above ground"] == (400, 500)
    assert idx["UGRD:10 m above ground"] == (500, 600)
    assert idx["VGRD:10 m above ground"] == (600, 700)
    # Last record: end == start (no next record); caller treats as EOF.
    assert idx["APCP:surface"] == (700, 700)


@responses.activate
def test_fetch_idx_handles_short_lines() -> None:
    """Lines with <6 fields are skipped (defensive parse)."""
    url = "https://example.s3.amazonaws.com/fake.idx"
    body = "1:0:d=20180601\n2:100:d=20180601:TMAX:2 m above ground:24 hour fcst\n"
    responses.add(responses.GET, url, body=body, status=200)
    idx = gfs._fetch_idx(url)
    assert list(idx) == ["TMAX:2 m above ground"]


# ----------------------------- _fetch_records (HTTP layer) -----------------------------


@responses.activate
def test_fetch_records_issues_range_request_and_returns_bytes() -> None:
    url = "https://example.s3.amazonaws.com/fake.grib2"
    # Mock-server doesn't enforce the Range header; we trust the request shape
    # is right (the test below inspects it explicitly).
    responses.add(responses.GET, url, body=b"GRIB" + b"\x00" * 100 + b"7777", status=206)

    out = gfs._fetch_records(url, [(0, 108)])

    assert out.startswith(b"GRIB")
    assert out.endswith(b"7777")
    assert len(responses.calls) == 1
    # Range header is start-(end_exclusive - 1).
    assert responses.calls[0].request.headers["Range"] == "bytes=0-107"


@responses.activate
def test_fetch_records_concatenates_multi_range() -> None:
    url = "https://example.s3.amazonaws.com/fake.grib2"
    responses.add(responses.GET, url, body=b"GRIB" + b"\xAA" * 10 + b"7777", status=206)
    responses.add(responses.GET, url, body=b"GRIB" + b"\xBB" * 10 + b"7777", status=206)

    out = gfs._fetch_records(url, [(0, 18), (100, 118)])

    assert len(responses.calls) == 2
    # The two mini-GRIBs are concatenated in order.
    assert out.count(b"GRIB") == 2
    assert out.count(b"7777") == 2


def test_fetch_records_empty_input_returns_empty() -> None:
    assert gfs._fetch_records("https://example/anything", []) == b""


# ----------------------------- _parse_grib_to_xarray (real fixture) -----------------------------


@pytest.mark.skipif(
    not (FIXTURE_DIR / "gefs_2018-06-01_t00z_f024_tmax.grib2").exists(),
    reason="GRIB fixture not present; see tests/fixtures/gfs/ docstring",
)
def test_parse_grib_real_fixture_tmax() -> None:
    """Parsing a saved real GEFS TMAX 2m mini-GRIB produces a sensible DataArray.

    The fixture is a single TMAX record from gefs.20180601/00/gec00.t00z.pgrb2af024
    (~43 KB). Sydney (-33.87, 151.21) at GEFS 1° = grid cell (-34, 151) — fall is
    on June 1; sensible TMAX is somewhere in 10..30 °C.
    """
    grib_bytes = (FIXTURE_DIR / "gefs_2018-06-01_t00z_f024_tmax.grib2").read_bytes()

    da = gfs._parse_grib_to_xarray(grib_bytes, "TMAX:2 m above ground")

    # cfgrib renames TMAX → tmax. We don't assert on the var name (varies
    # across cfgrib versions) but the shape should be (lat, lon).
    assert "latitude" in da.dims
    assert "longitude" in da.dims
    # 1° global grid → 181 lats, 360 lons.
    assert da.sizes["latitude"] == 181
    assert da.sizes["longitude"] == 360

    # Sample Sydney area. GEFS 1° lats descend 90..-90; longitudes 0..359.
    syd = da.sel(latitude=-34.0, longitude=151.0, method="nearest")
    val_k = float(syd)  # Kelvin
    # June in Sydney: daily max ~14-22°C → 287-295 K. Allow a wide band
    # for forecast scatter and the f024 18-24h max window.
    assert 270.0 < val_k < 310.0, f"TMAX {val_k} K out of plausible range"


def test_parse_grib_rejects_non_grib_buffer() -> None:
    with pytest.raises(RuntimeError, match="GRIB"):
        gfs._parse_grib_to_xarray(b"not a grib", "TMAX:2 m above ground")


# ============================ Session 2: multi-horizon + grid parquet ============================


# ----------------------------- _leads_for_horizon -----------------------------


def test_leads_for_horizon_day_1() -> None:
    """Horizon 1 spans the 4 6-hour leads of the first forecast day."""
    assert gfs._leads_for_horizon(1) == [6, 12, 18, 24]


def test_leads_for_horizon_day_7() -> None:
    """Horizon 7 spans leads 150..168 (4 records, 6h apart)."""
    assert gfs._leads_for_horizon(7) == [150, 156, 162, 168]


def test_leads_for_horizon_rejects_zero() -> None:
    with pytest.raises(ValueError, match="horizon"):
        gfs._leads_for_horizon(0)


# ----------------------------- _grid_parquet_path -----------------------------


def test_grid_parquet_path_format() -> None:
    """Per-(date, horizon) output filenames follow `<YYYY-MM-DD>_h<N>.parquet`."""
    p = gfs._grid_parquet_path(Path("/tmp/out"), dt.date(2024, 3, 15), 3)
    assert p.name == "2024-03-15_h3.parquet"


# ----------------------------- _open_grib_dataset (multi-record) -----------------------------


FIXTURE_VARS = ("tmax", "tmin", "apcp", "ugrd10m", "vgrd10m")


def _load_concat_fixture() -> bytes:
    """Concatenate the 5 single-variable GEFS fixtures into one buffer.

    This mimics what `_fetch_records` produces when called with the 5
    needed byte ranges from a live `.idx` — a buffer of 5 mini-GRIBs.
    """
    out = b""
    for v in FIXTURE_VARS:
        fp = FIXTURE_DIR / f"gefs_2018-06-01_t00z_f024_{v}.grib2"
        out += fp.read_bytes()
    return out


@pytest.mark.skipif(
    not all(
        (FIXTURE_DIR / f"gefs_2018-06-01_t00z_f024_{v}.grib2").exists()
        for v in FIXTURE_VARS
    ),
    reason="GRIB fixtures not present; see tests/fixtures/gfs/ docstring",
)
def test_open_grib_dataset_merges_five_variables() -> None:
    """Multi-record buffer (TMAX+TMIN+APCP+U10+V10) parses to one merged Dataset."""
    ds = gfs._open_grib_dataset(_load_concat_fixture())

    # All 5 variables present after cfgrib's rename.
    for name in ("tmax", "tmin", "tp", "u10", "v10"):
        assert name in ds.data_vars, f"missing {name} in {list(ds.data_vars)}"
    # Lat/lon are the merge axes — GEFS 1° global = 181 × 360.
    assert ds.sizes["latitude"] == 181
    assert ds.sizes["longitude"] == 360


# ----------------------------- _select_nsw_box -----------------------------


@pytest.mark.skipif(
    not (FIXTURE_DIR / "gefs_2018-06-01_t00z_f024_tmax.grib2").exists(),
    reason="GRIB fixture not present",
)
def test_select_nsw_box_subsets_to_bbox() -> None:
    """NSW slice yields 10 lats × 14-15 lons at GEFS 1° resolution."""
    ds = gfs._open_grib_dataset(_load_concat_fixture())
    nsw = gfs._select_nsw_box(ds["tmax"])
    assert nsw.sizes["latitude"] == 10
    # 140.5..154.0 step 1° → 14 cells starting at 141.
    assert nsw.sizes["longitude"] == 14
    # First lat at the northern edge (closest to -28); last at southern.
    assert float(nsw.latitude[0]) == -28.0
    assert float(nsw.latitude[-1]) == -37.0


# ----------------------------- Aggregation helpers -----------------------------


def test_aggregate_per_variable_max_min_sum() -> None:
    """Reducer applies correctly across the lead dimension."""
    arr_a = xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]]), dims=("latitude", "longitude"))
    arr_b = xr.DataArray(np.array([[5.0, 1.0], [2.0, 7.0]]), dims=("latitude", "longitude"))
    per_lead = {6: arr_a, 12: arr_b}
    assert gfs._aggregate_per_variable(per_lead, "max").values.tolist() == [[5.0, 2.0], [3.0, 7.0]]
    assert gfs._aggregate_per_variable(per_lead, "min").values.tolist() == [[1.0, 1.0], [2.0, 4.0]]
    assert gfs._aggregate_per_variable(per_lead, "sum").values.tolist() == [[6.0, 3.0], [5.0, 11.0]]


def test_aggregate_per_variable_rejects_unknown_reducer() -> None:
    arr = xr.DataArray(np.array([1.0]), dims=("x",))
    with pytest.raises(ValueError, match="reducer"):
        gfs._aggregate_per_variable({0: arr}, "median")


def test_aggregate_wind_speed_max_combines_u_and_v() -> None:
    """U/V → speed = sqrt(U²+V²) per lead, then max across leads."""
    # Lead 6: U=3, V=4 → speed=5.  Lead 12: U=6, V=8 → speed=10.
    u = {
        6: xr.DataArray(np.array([[3.0]]), dims=("latitude", "longitude")),
        12: xr.DataArray(np.array([[6.0]]), dims=("latitude", "longitude")),
    }
    v = {
        6: xr.DataArray(np.array([[4.0]]), dims=("latitude", "longitude")),
        12: xr.DataArray(np.array([[8.0]]), dims=("latitude", "longitude")),
    }
    out = gfs._aggregate_wind_speed_max(u, v)
    assert float(out.values[0, 0]) == pytest.approx(10.0)


def test_aggregate_wind_speed_max_rejects_lead_mismatch() -> None:
    arr = xr.DataArray(np.array([[1.0]]), dims=("latitude", "longitude"))
    with pytest.raises(ValueError, match="mismatch"):
        gfs._aggregate_wind_speed_max({6: arr}, {12: arr})


# ----------------------------- End-to-end: fetch_and_write_one_day -----------------------------


@pytest.fixture()
def fixture_dataset():
    """Load the 5-variable GEFS 2018-06-01 f024 fixture as a merged Dataset."""
    if not all(
        (FIXTURE_DIR / f"gefs_2018-06-01_t00z_f024_{v}.grib2").exists()
        for v in FIXTURE_VARS
    ):
        pytest.skip("GRIB fixtures not present")
    return gfs._open_grib_dataset(_load_concat_fixture())


def test_fetch_and_write_one_day_single_horizon_e2e(tmp_path: Path, fixture_dataset) -> None:
    """Full pipeline: mock the network layer with the fixture, drive
    fetch_and_write_one_day for horizon 1, then read back the parquet and
    sanity-check schema + value ranges."""

    def _mock_fetch_lead(date: dt.date, cycle: str, lead_h: int, resolution: str):
        # Same fixture for all 4 leads — exercises the multi-lead aggregation
        # path even though the fixture only covers f024.
        return fixture_dataset

    with patch.object(gfs, "_fetch_lead", side_effect=_mock_fetch_lead):
        paths = gfs.fetch_and_write_one_day(
            dt.date(2018, 6, 1), tmp_path, horizons=(1,), cycle="00",
        )

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].name == "2018-06-01_h1.parquet"

    df = pd.read_parquet(paths[0])
    # 10 lats × 14 lons = 140 cells.
    assert len(df) == 140
    assert list(df.columns) == [
        "lat_idx", "lon_idx", "lat", "lon",
        "wx_temp_max_c", "wx_temp_min_c",
        "wx_precipitation_mm", "wx_wind_speed_max_kmh",
    ]
    # June Sydney TMAX (K→C): broad sanity band.
    assert -10.0 < df.wx_temp_max_c.min() < df.wx_temp_max_c.max() < 35.0
    # Wind km/h: positive, bounded — fixture had max ~67 km/h over NSW.
    assert df.wx_wind_speed_max_kmh.min() >= 0
    assert df.wx_wind_speed_max_kmh.max() < 200.0
    # Latitudes are negative (southern hemisphere).
    assert (df["lat"] < 0).all()


def test_fetch_and_write_one_day_multi_horizon_writes_one_per_horizon(
    tmp_path: Path, fixture_dataset,
) -> None:
    """Multi-horizon: 3 horizons → 3 output parquets, each properly named."""
    with patch.object(gfs, "_fetch_lead", side_effect=lambda *a, **kw: fixture_dataset):
        paths = gfs.fetch_and_write_one_day(
            dt.date(2018, 6, 1), tmp_path, horizons=(1, 2, 3), cycle="00",
        )

    assert len(paths) == 3
    expected_names = {"2018-06-01_h1.parquet", "2018-06-01_h2.parquet", "2018-06-01_h3.parquet"}
    assert {p.name for p in paths} == expected_names
    # Each parquet has the same shape (same fixture → same cells).
    for p in paths:
        df = pd.read_parquet(p)
        assert len(df) == 140


def test_fetch_and_write_one_day_no_tmp_leftover(tmp_path: Path, fixture_dataset) -> None:
    """Atomic write: after success, no `.tmp` files remain in the output dir."""
    with patch.object(gfs, "_fetch_lead", side_effect=lambda *a, **kw: fixture_dataset):
        gfs.fetch_and_write_one_day(
            dt.date(2018, 6, 1), tmp_path, horizons=(1, 2), cycle="00",
        )
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"unexpected .tmp leftovers: {leftovers}"


def test_fetch_and_write_one_day_cache_hit_skips_network(
    tmp_path: Path, fixture_dataset,
) -> None:
    """If all per-(date, horizon) parquets exist and `force` is False, no fetch."""
    # First call populates the cache.
    with patch.object(gfs, "_fetch_lead", side_effect=lambda *a, **kw: fixture_dataset):
        gfs.fetch_and_write_one_day(
            dt.date(2018, 6, 1), tmp_path, horizons=(1,), cycle="00",
        )
    # Second call should NOT call _fetch_lead — assert via call count.
    with patch.object(gfs, "_fetch_lead") as m:
        paths = gfs.fetch_and_write_one_day(
            dt.date(2018, 6, 1), tmp_path, horizons=(1,), cycle="00", force=False,
        )
        m.assert_not_called()
    assert len(paths) == 1


def test_fetch_and_write_one_day_force_refetches(
    tmp_path: Path, fixture_dataset,
) -> None:
    """`force=True` overrides the cache and re-fetches every horizon."""
    with patch.object(gfs, "_fetch_lead", side_effect=lambda *a, **kw: fixture_dataset):
        gfs.fetch_and_write_one_day(
            dt.date(2018, 6, 1), tmp_path, horizons=(1,), cycle="00",
        )
    with patch.object(
        gfs, "_fetch_lead", side_effect=lambda *a, **kw: fixture_dataset,
    ) as m:
        gfs.fetch_and_write_one_day(
            dt.date(2018, 6, 1), tmp_path, horizons=(1,), cycle="00", force=True,
        )
        # 4 leads for horizon 1 → 4 fetch calls when force=True.
        assert m.call_count == 4


# ----------------------------- fetch (date range) -----------------------------


def test_fetch_range_iterates_dates_and_skips_cached(
    tmp_path: Path, fixture_dataset,
) -> None:
    """`fetch()` walks the date range; pre-cached dates short-circuit the per-day path."""
    # Seed the cache for 2018-06-01 only.
    (tmp_path / "2018-06-01_h1.parquet").touch()

    with patch.object(
        gfs, "fetch_and_write_one_day",
        side_effect=lambda date, out_dir, **kw: [
            out_dir / f"{date.isoformat()}_h1.parquet",
        ],
    ) as m:
        gfs.fetch(
            start="2018-06-01", end="2018-06-03",
            out_dir=tmp_path, horizons=(1,), cycle="00",
        )
        # 2 of the 3 days need a fetch (2018-06-02 and 2018-06-03).
        # 2018-06-01 is cached at the outer loop and skipped.
        assert m.call_count == 2


def test_fetch_range_rejects_inverted_dates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be <="):
        gfs.fetch(start="2024-12-31", end="2024-01-01", out_dir=tmp_path)
