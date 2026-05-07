"""Hermetic tests for spatial.resolve_addrs.

We don't go anywhere near the real G-NAF S3 bucket or Nominatim — both
geocoders are injected via factories that return canned results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from fuel_pred.spatial import resolve_addrs as ra


@dataclass
class FakeResult:
    """Stand-in for census_augment.geocoding.*.GeocodeResult."""

    lat: float | None
    lon: float | None
    mb_code: str | None = None

    @property
    def is_success(self) -> bool:
        return self.lat is not None and self.lon is not None


class FakeGeocoder:
    """Records the addresses it sees and returns a canned result for each."""

    def __init__(self, responses: dict[str, FakeResult]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def geocode(self, address: str) -> FakeResult:
        self.calls.append(address)
        return self.responses.get(address, FakeResult(None, None))


@pytest.fixture
def stations_in(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "station_id": "s1",
                "name": "BP Mascot",
                "address": "1 Botany Rd",
                "suburb": "Mascot",
                "postcode": "2020",
            },
            {
                "station_id": "s2",
                "name": "7-Eleven Newtown",
                "address": "100 King St",
                "suburb": "Newtown",
                "postcode": "2042",
            },
        ]
    )
    p = tmp_path / "stations_in.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    return p


def _addr(suburb: str, postcode: str, address: str) -> str:
    return f"{address}, {suburb}, NSW {postcode}, Australia"


def test_gnaf_resolves_all_addresses(tmp_path: Path, stations_in: Path) -> None:
    addr_s1 = _addr("Mascot", "2020", "1 Botany Rd")
    addr_s2 = _addr("Newtown", "2042", "100 King St")
    gnaf = FakeGeocoder(
        {
            addr_s1: FakeResult(-33.93, 151.20, mb_code="MB1"),
            addr_s2: FakeResult(-33.90, 151.18, mb_code="MB2"),
        }
    )
    nom = FakeGeocoder({})

    out = tmp_path / "stations_out.parquet"
    ra.resolve(
        stations_in,
        out,
        cache_dir=tmp_path / "cache",
        gnaf_factory=lambda: gnaf,
        nominatim_factory=lambda: nom,
    )

    df = pd.read_parquet(out)
    assert (df["geocoder"] == "gnaf").all()
    assert df.loc[df["station_id"] == "s1", "lat"].iloc[0] == pytest.approx(-33.93)
    assert df.loc[df["station_id"] == "s1", "mb_code"].iloc[0] == "MB1"
    assert nom.calls == []


def test_nominatim_picks_up_gnaf_misses(tmp_path: Path, stations_in: Path) -> None:
    addr_s1 = _addr("Mascot", "2020", "1 Botany Rd")
    addr_s2 = _addr("Newtown", "2042", "100 King St")
    gnaf = FakeGeocoder(
        {
            addr_s1: FakeResult(-33.93, 151.20, mb_code="MB1"),
            # s2 missing → G-NAF returns FakeResult(None, None) → fallback.
        }
    )
    nom = FakeGeocoder(
        {
            addr_s2: FakeResult(-33.90, 151.18),
        }
    )

    out = tmp_path / "stations_out.parquet"
    ra.resolve(
        stations_in,
        out,
        cache_dir=tmp_path / "cache",
        gnaf_factory=lambda: gnaf,
        nominatim_factory=lambda: nom,
    )

    df = pd.read_parquet(out)
    s1 = df[df["station_id"] == "s1"].iloc[0]
    s2 = df[df["station_id"] == "s2"].iloc[0]
    assert s1["geocoder"] == "gnaf"
    assert s2["geocoder"] == "nominatim"
    # Nominatim was only called for the G-NAF miss.
    assert nom.calls == [addr_s2]


def test_idempotent_skip_when_already_geocoded(tmp_path: Path) -> None:
    """A second run on a fully-resolved file should call neither geocoder."""
    df = pd.DataFrame(
        [
            {
                "station_id": "s1",
                "name": "BP Mascot",
                "address": "1 Botany Rd",
                "suburb": "Mascot",
                "postcode": "2020",
                "lat": -33.93,
                "lon": 151.20,
                "geocoder": "gnaf",
                "mb_code": "MB1",
            }
        ]
    )
    p = tmp_path / "stations.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)

    gnaf = FakeGeocoder({})
    nom = FakeGeocoder({})

    ra.resolve(
        p,
        p,
        cache_dir=tmp_path / "cache",
        gnaf_factory=lambda: gnaf,
        nominatim_factory=lambda: nom,
    )

    assert gnaf.calls == []
    assert nom.calls == []


def test_force_re_resolves_every_row(tmp_path: Path, stations_in: Path) -> None:
    """``force=True`` ignores existing lat/lon and re-runs every row."""
    df = pd.read_parquet(stations_in)
    df["lat"] = [-33.0, -33.1]
    df["lon"] = [151.0, 151.1]
    df["geocoder"] = "stale"
    df["mb_code"] = pd.NA
    df.to_parquet(stations_in, engine="pyarrow", compression="zstd", index=False)

    addr_s1 = _addr("Mascot", "2020", "1 Botany Rd")
    addr_s2 = _addr("Newtown", "2042", "100 King St")
    gnaf = FakeGeocoder(
        {
            addr_s1: FakeResult(-33.93, 151.20, mb_code="MB1"),
            addr_s2: FakeResult(-33.90, 151.18, mb_code="MB2"),
        }
    )
    nom = FakeGeocoder({})

    out = tmp_path / "stations_out.parquet"
    ra.resolve(
        stations_in,
        out,
        cache_dir=tmp_path / "cache",
        force=True,
        gnaf_factory=lambda: gnaf,
        nominatim_factory=lambda: nom,
    )

    df = pd.read_parquet(out)
    assert sorted(df["geocoder"].unique()) == ["gnaf"]
    assert df.loc[df["station_id"] == "s1", "lat"].iloc[0] == pytest.approx(-33.93)


def test_partial_resume_only_resolves_missing(tmp_path: Path, stations_in: Path) -> None:
    """One row already resolved → only the other gets a geocoder call."""
    df = pd.read_parquet(stations_in)
    df["lat"] = [-33.93, None]
    df["lon"] = [151.20, None]
    df["geocoder"] = ["gnaf", None]
    df["mb_code"] = ["MB1", None]
    df.to_parquet(stations_in, engine="pyarrow", compression="zstd", index=False)

    addr_s2 = _addr("Newtown", "2042", "100 King St")
    gnaf = FakeGeocoder({addr_s2: FakeResult(-33.90, 151.18, mb_code="MB2")})
    nom = FakeGeocoder({})

    out = tmp_path / "stations_out.parquet"
    ra.resolve(
        stations_in,
        out,
        cache_dir=tmp_path / "cache",
        gnaf_factory=lambda: gnaf,
        nominatim_factory=lambda: nom,
    )

    # G-NAF was called exactly once — for s2.
    assert gnaf.calls == [addr_s2]
    df = pd.read_parquet(out)
    assert df.loc[df["station_id"] == "s2", "mb_code"].iloc[0] == "MB2"


def test_failure_leaves_lat_lon_null(tmp_path: Path, stations_in: Path) -> None:
    gnaf = FakeGeocoder({})  # everything misses
    nom = FakeGeocoder({})

    out = tmp_path / "stations_out.parquet"
    ra.resolve(
        stations_in,
        out,
        cache_dir=tmp_path / "cache",
        gnaf_factory=lambda: gnaf,
        nominatim_factory=lambda: nom,
    )

    df = pd.read_parquet(out)
    assert df["lat"].isna().all()
    assert df["geocoder"].isna().all()


def test_format_address_strips_blanks() -> None:
    row = pd.Series(
        {"address": "1 Main St ", "suburb": "Foo", "postcode": "2000", "name": "X"}
    )
    assert ra._format_address(row) == "1 Main St, Foo, NSW 2000, Australia"
    # Missing suburb shouldn't insert a stray comma.
    row2 = pd.Series({"address": "1 Main St", "suburb": "", "postcode": "2000", "name": "X"})
    out = ra._format_address(row2)
    assert ", , " not in out


def test_in_place_overwrite_is_atomic(tmp_path: Path, stations_in: Path) -> None:
    """If the writer crashed, no half-written parquet should overwrite the input."""
    addr_s1 = _addr("Mascot", "2020", "1 Botany Rd")
    addr_s2 = _addr("Newtown", "2042", "100 King St")
    gnaf = FakeGeocoder(
        {
            addr_s1: FakeResult(-33.93, 151.20, mb_code="MB1"),
            addr_s2: FakeResult(-33.90, 151.18, mb_code="MB2"),
        }
    )
    nom = FakeGeocoder({})

    # Use the same path for input and output (the common case).
    ra.resolve(
        stations_in,
        stations_in,
        cache_dir=tmp_path / "cache",
        gnaf_factory=lambda: gnaf,
        nominatim_factory=lambda: nom,
    )

    df = pd.read_parquet(stations_in)
    assert "lat" in df.columns
    # No `.tmp` file left behind.
    assert not (stations_in.parent / (stations_in.name + ".tmp")).exists()
