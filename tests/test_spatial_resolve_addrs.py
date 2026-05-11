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
    """Mirror what `_format_address` produces from FuelCheck's columns.

    The street-only fixture inputs (no postcode tail in the `address`
    field) trigger the suburb + postcode appending path. No state is
    appended (FuelCheck's address column carries it when present); no
    `, Australia` suffix (it broke the augmentor's parser — see
    PR for context).
    """
    return f"{address}, {suburb}, {postcode}"


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


def test_format_address_passes_through_canonical_full_address() -> None:
    """The 99.9% case: address already ends with a 4-digit postcode →
    return as-is so the augmentor's `parse_address()` can extract
    locality/state/postcode cleanly. Adding ANY decoration breaks the
    parser (Tier 2/3 then have no postcode and silently miss).
    """
    row = pd.Series(
        {
            "address": "1 Braidwood Rd, GOULBURN NSW 2580",
            "suburb": "GOULBURN",
            "postcode": "2580",
            "name": "X",
        }
    )
    assert ra._format_address(row) == "1 Braidwood Rd, GOULBURN NSW 2580"


def test_format_address_appends_missing_suburb_and_postcode() -> None:
    """Street-only address column: suburb + postcode appended (no `, Australia`)."""
    row = pd.Series(
        {"address": "1 Main St ", "suburb": "Foo", "postcode": "2000", "name": "X"}
    )
    # Cross-street stripping leaves the trailing space stripped too.
    assert ra._format_address(row) == "1 Main St, Foo, 2000"


def test_format_address_never_appends_australia_suffix() -> None:
    """The `, Australia` suffix corrupts the augmentor's parser — it gets
    swept into `locality` and the parser fails to extract postcode.
    Regression guard for the bug that produced 0% G-NAF hit rate.
    """
    canonical = pd.Series(
        {"address": "1 Main St, Foo NSW 2000", "suburb": "Foo", "postcode": "2000", "name": "X"}
    )
    street_only = pd.Series(
        {"address": "1 Main St", "suburb": "Foo", "postcode": "2000", "name": "X"}
    )
    assert "Australia" not in ra._format_address(canonical)
    assert "Australia" not in ra._format_address(street_only)


def test_format_address_preserves_non_nsw_state_in_address_column() -> None:
    """ACT stations carry their state in the address column — don't override."""
    row = pd.Series(
        {
            "address": "41 Federal Hwy, Lyneham ACT 2602",
            "suburb": "Lyneham",
            "postcode": "2602",
            "name": "X",
        }
    )
    out = ra._format_address(row)
    assert "ACT" in out
    assert "NSW" not in out


def test_format_address_strips_blanks() -> None:
    """Missing suburb shouldn't insert a stray double-comma."""
    row = pd.Series({"address": "1 Main St", "suburb": "", "postcode": "2000", "name": "X"})
    out = ra._format_address(row)
    assert ", , " not in out


def test_format_address_handles_postcode_as_float_artifact() -> None:
    """`clean.fuelcheck` sometimes emits postcodes as floats (`'2776.0'`).

    The street-only path was previously appending the literal `'2776.0'`
    to the address (since `'2776.0'` isn't a substring of an address
    containing `2776`). Normalised postcode now lands as `'2776'`.
    """
    row = pd.Series(
        {"address": "1 Main St", "suburb": "Foo", "postcode": "2776.0", "name": "X"}
    )
    assert ra._format_address(row) == "1 Main St, Foo, 2776"


def test_format_address_canonical_address_with_float_postcode_passes_through() -> None:
    """Float-postcode + canonical address → passthrough wins. The address
    field already has the 4-digit postcode at the end; the float column
    is ignored thanks to the trailing-postcode short-circuit.
    """
    row = pd.Series(
        {
            "address": "450 Great Western Hw, Faulconbridge NSW 2776",
            "suburb": "Faulconbridge",
            "postcode": "2776.0",
            "name": "X",
        }
    )
    out = ra._format_address(row)
    assert out == "450 Great Western Hw, Faulconbridge NSW 2776"
    assert "2776.0" not in out


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2776", "2776"),
        ("2776.0", "2776"),
        ("  2776  ", "2776"),
        (2776.0, "2776"),  # numeric input from a non-stringified column
        (2776, "2776"),
        ("", ""),
        (None, ""),
        ("LPO 1234", "LPO 1234"),  # non-numeric postcode-ish strings: untouched
    ],
)
def test_normalise_postcode(raw: object, expected: str) -> None:
    """Strip the spurious trailing `.0` that pandas can introduce."""
    assert ra._normalise_postcode(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("78 Great Western Hwy Cnr Ross St", "78 Great Western Hwy"),
        ("78 Great Western Hwy CNR Ross St", "78 Great Western Hwy"),
        ("100 King St corner of Smith St", "100 King St"),
        ("1 Main St c/o Service Rd", "1 Main St"),
        ("1 Main St", "1 Main St"),  # no annotation — passthrough
    ],
)
def test_clean_street_strips_cross_street_annotations(raw: str, expected: str) -> None:
    assert ra._clean_street(raw) == expected


def test_format_address_drops_cross_street_in_full_pipeline() -> None:
    """End-to-end: cross-street annotation in the address column gets stripped.

    Without trailing postcode in the address (the cross-street was the
    whole tail), suburb + postcode are appended. No `, Australia`.
    """
    row = pd.Series(
        {
            "address": "78 Great Western Hwy Cnr Ross St",
            "suburb": "Glenbrook",
            "postcode": "2773",
            "name": "Ampol Foodary",
        }
    )
    assert ra._format_address(row) == "78 Great Western Hwy, Glenbrook, 2773"


def test_format_address_drops_cross_street_with_canonical_tail() -> None:
    """Cross-street + canonical postcode tail: strip the cnr, then the
    trailing-postcode short-circuit fires (suburb/postcode already in
    the address text)."""
    row = pd.Series(
        {
            "address": "78 Great Western Hwy Cnr Ross St, Glenbrook NSW 2773",
            "suburb": "Glenbrook",
            "postcode": "2773",
            "name": "Ampol Foodary",
        }
    )
    assert ra._format_address(row) == "78 Great Western Hwy, Glenbrook NSW 2773"


class _BrokenGnaf:
    """A G-NAF stub that always raises — simulates the upstream issue #8."""

    def geocode(self, address: str) -> FakeResult:  # pragma: no cover - never returns
        raise RuntimeError("G-NAF remote view is missing required columns")


def test_falls_back_to_nominatim_when_gnaf_warmup_fails(
    tmp_path: Path,
    stations_in: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When G-NAF init fails, the run continues with Nominatim-only."""
    addr_s1 = _addr("Mascot", "2020", "1 Botany Rd")
    addr_s2 = _addr("Newtown", "2042", "100 King St")
    nom = FakeGeocoder(
        {
            addr_s1: FakeResult(-33.93, 151.20),
            addr_s2: FakeResult(-33.90, 151.18),
        }
    )

    # _try_gnaf_warmup only triggers when gnaf_factory is None — so we
    # patch the module-level constructor instead.
    import fuel_pred.spatial.resolve_addrs as ra_mod

    monkeypatch_built = ra_mod._build_geocoders

    def fake_build(*args: object, **kwargs: object) -> tuple[object, object]:
        return _BrokenGnaf(), nom

    ra_mod._build_geocoders = fake_build  # type: ignore[assignment]
    try:
        out = tmp_path / "stations_out.parquet"
        with caplog.at_level("WARNING", logger="fuel_pred.spatial.resolve_addrs"):
            ra.resolve(stations_in, out, cache_dir=tmp_path / "cache")
    finally:
        ra_mod._build_geocoders = monkeypatch_built  # type: ignore[assignment]

    df = pd.read_parquet(out)
    assert (df["geocoder"] == "nominatim").all()
    assert any("G-NAF init failed" in rec.message for rec in caplog.records)


# ----------------------------- progress logging -----------------------------


def test_format_eta_seconds_only() -> None:
    assert ra._format_eta(45) == "45s"


def test_format_eta_minutes_and_seconds() -> None:
    assert ra._format_eta(125) == "2m 5s"


def test_format_eta_hours_minutes_seconds() -> None:
    assert ra._format_eta(3 * 3600 + 12 * 60 + 8) == "3h 12m 8s"


def test_format_eta_handles_zero_or_negative() -> None:
    assert ra._format_eta(0) == "?"
    assert ra._format_eta(-1) == "?"


def test_progress_logger_emits_at_count_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reaching PROGRESS_LOG_INTERVAL_COUNT items triggers a log even
    if the time threshold hasn't elapsed."""
    progress = ra._ProgressLogger(total=1000)
    with caplog.at_level("INFO", logger="fuel_pred.spatial.resolve_addrs"):
        # Below threshold — silent.
        for i in range(1, ra.PROGRESS_LOG_INTERVAL_COUNT):
            progress.maybe_emit(i, gnaf_hits=i, nominatim_hits=0, failures=0)
        assert not any("geocoding progress" in r.message for r in caplog.records)
        # Hit the threshold — one log.
        progress.maybe_emit(
            ra.PROGRESS_LOG_INTERVAL_COUNT,
            gnaf_hits=ra.PROGRESS_LOG_INTERVAL_COUNT,
            nominatim_hits=0,
            failures=0,
        )
    progress_msgs = [r for r in caplog.records if "geocoding progress" in r.message]
    assert len(progress_msgs) == 1
    msg = progress_msgs[0].message
    assert f"{ra.PROGRESS_LOG_INTERVAL_COUNT}/1000" in msg
    assert "gnaf=" in msg and "eta " in msg


def test_progress_logger_emits_at_time_threshold(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A slow run that hasn't hit the count threshold still gets a log
    once the time threshold elapses."""
    fake_now = [1000.0]
    monkeypatch.setattr(ra.time, "monotonic", lambda: fake_now[0])

    progress = ra._ProgressLogger(total=1000)
    with caplog.at_level("INFO", logger="fuel_pred.spatial.resolve_addrs"):
        # Process just 5 items but advance the clock past the time threshold.
        progress.maybe_emit(5, gnaf_hits=5, nominatim_hits=0, failures=0)
        assert not any("geocoding progress" in r.message for r in caplog.records)
        fake_now[0] += ra.PROGRESS_LOG_INTERVAL_SECONDS + 0.1
        progress.maybe_emit(6, gnaf_hits=6, nominatim_hits=0, failures=0)

    progress_msgs = [r for r in caplog.records if "geocoding progress" in r.message]
    assert len(progress_msgs) == 1
    assert "6/1000" in progress_msgs[0].message


def test_progress_logger_silent_below_both_thresholds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No log emitted while both count and time deltas are sub-threshold."""
    fake_now = [0.0]
    monkeypatch.setattr(ra.time, "monotonic", lambda: fake_now[0])

    progress = ra._ProgressLogger(total=10000)
    with caplog.at_level("INFO", logger="fuel_pred.spatial.resolve_addrs"):
        for i in range(1, 50):  # well under PROGRESS_LOG_INTERVAL_COUNT
            fake_now[0] += 0.01  # 0.01s/iter → well under PROGRESS_LOG_INTERVAL_SECONDS
            progress.maybe_emit(i, gnaf_hits=i, nominatim_hits=0, failures=0)
    assert not any("geocoding progress" in r.message for r in caplog.records)


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
