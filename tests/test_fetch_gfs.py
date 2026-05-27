"""Hermetic tests for fetch.gfs — URL routing, idx parsing, byte-range fetch, GRIB parse.

Real-network smoke is in `tools/` (separate, opt-in). These tests use:
  - small synthetic .idx text for parser tests
  - the saved real-GRIB fixtures under `tests/fixtures/gfs/` for parse tests
    (each <100 KB; one-time fetch from the live S3 bucket).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import responses

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
