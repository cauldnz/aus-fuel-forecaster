"""Hermetic tests for build.enrich_census.

The augmentor's `Pipeline` is replaced with a stub that returns a
deterministic enriched frame — no S3, no boundaries download.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from fuel_pred.build import enrich_census as ec


@dataclass
class _StubResult:
    df: pd.DataFrame


class _StubPipeline:
    """Minimal stand-in for `census_augment.Pipeline.create(...)`."""

    def __init__(self, sa2_lookup: dict[tuple[float, float], dict[str, object]]) -> None:
        self.sa2_lookup = sa2_lookup
        self.calls: list[pd.DataFrame] = []

    def augment(
        self,
        df: pd.DataFrame,
        *,
        latitude_column: str,
        longitude_column: str,
    ) -> _StubResult:
        self.calls.append(df.copy())
        out = df.copy()
        added_cols = (
            "sa2_code",
            "sa2_name",
            "sa2_total_population",
            "sa2_median_age",
            "sa2_median_household_income_weekly",
            "sa2_pct_drive_to_work",
            "sa2_motor_vehicles_per_dwelling",
            "sa2_pct_renters",
            "sa2_pct_employed_full_time",
            "sa2_pct_aged_65_plus",
            "sa2_pct_one_parent_family",
        )
        for col in added_cols:
            out[col] = pd.NA
        for idx, row in df.iterrows():
            key = (round(row[latitude_column], 4), round(row[longitude_column], 4))
            data = self.sa2_lookup.get(key)
            if data is not None:
                for col, val in data.items():
                    out.at[idx, col] = val
        return _StubResult(df=out)


def _make_stub_factory(sa2_lookup: dict[tuple[float, float], dict[str, object]]):
    def factory() -> _StubPipeline:
        return _StubPipeline(sa2_lookup)

    return factory


@pytest.fixture
def stations_in(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "station_id": "s1",
                "name": "BP Mascot",
                "lat": -33.93,
                "lon": 151.20,
            },
            {
                "station_id": "s2",
                "name": "EG Ampol Newport",
                "lat": -33.65,
                "lon": 151.32,
            },
        ]
    )
    p = tmp_path / "stations.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    return p


@pytest.fixture
def seifa_loader():
    """A test seam returning a synthetic SEIFA frame in the augmentor's
    native shape (indexed by `sa2_code_2021`).
    """
    def _loader() -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "sa2_code_2021": ["117011635", "122021422", "999999999"],
                "irsd_score": [1098.0, 1102.0, 999.0],
                "irsd_aus_decile": [9.0, 10.0, 5.0],
            }
        ).set_index("sa2_code_2021")
        return df

    return _loader


@pytest.fixture
def seifa_cache_dir(tmp_path: Path) -> Path:
    """Cache dir; not actually written to in tests because seifa_loader
    short-circuits the SeifaDataSource construction."""
    return tmp_path / "seifa_cache"


@pytest.fixture
def lookup() -> dict[tuple[float, float], dict[str, object]]:
    return {
        (-33.93, 151.20): {
            "sa2_code": "117011635",
            "sa2_name": "Mascot",
            "sa2_total_population": 21573,
            "sa2_median_age": 30,
            "sa2_median_household_income_weekly": 1900,
            # PRESET derivations (augmentor v1.4.2+).
            "sa2_pct_drive_to_work": 38.5,
            "sa2_motor_vehicles_per_dwelling": 0.62,
            "sa2_pct_renters": 65.85,
            "sa2_pct_employed_full_time": 63.0,
            "sa2_pct_aged_65_plus": 14.82,
            "sa2_pct_one_parent_family": 18.0,
        },
        (-33.65, 151.32): {
            "sa2_code": "122021422",
            "sa2_name": "Newport - Bilgola",
            "sa2_total_population": 13681,
            "sa2_median_age": 46,
            "sa2_median_household_income_weekly": 2400,
            "sa2_pct_drive_to_work": 47.2,
            "sa2_motor_vehicles_per_dwelling": 1.85,
            "sa2_pct_renters": 22.4,
            "sa2_pct_employed_full_time": 58.1,
            "sa2_pct_aged_65_plus": 24.8,
            "sa2_pct_one_parent_family": 8.3,
        },
    }


def test_writes_full_schema(
    tmp_path: Path,
    stations_in: Path,
    seifa_cache_dir: Path,
    seifa_loader,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    out = tmp_path / "stations_out.parquet"
    ec.enrich(
        stations_in,
        out,
        seifa_cache_dir=seifa_cache_dir,
        seifa_loader=seifa_loader,
        pipeline_factory=_make_stub_factory(lookup),
    )

    df = pd.read_parquet(out)
    for col in ec.ENRICHED_COLUMNS:
        assert col in df.columns

    # Both stations enriched.
    assert df["sa2_code"].notna().all()
    assert df["sa2_total_population"].notna().all()
    assert df["sa2_seifa_irsd_score"].notna().all()


def test_seifa_irsd_joins_correctly(
    tmp_path: Path,
    stations_in: Path,
    seifa_cache_dir: Path,
    seifa_loader,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    out = tmp_path / "stations_out.parquet"
    ec.enrich(
        stations_in,
        out,
        seifa_cache_dir=seifa_cache_dir,
        seifa_loader=seifa_loader,
        pipeline_factory=_make_stub_factory(lookup),
    )
    df = pd.read_parquet(out)
    by_id = {row["station_id"]: row for _, row in df.iterrows()}
    assert int(by_id["s1"]["sa2_seifa_irsd_score"]) == 1098
    assert int(by_id["s2"]["sa2_seifa_irsd_score"]) == 1102


def test_six_preset_derivations_populate(
    tmp_path: Path,
    stations_in: Path,
    seifa_cache_dir: Path,
    seifa_loader,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    """Augmentor v1.4.2 — the 6 PRESET-derived percentages populate per row.

    Previously these were stubbed null per spec §7.7.1; v1.4.2's PRESETs
    fixed against the real GCP DataPack let us pass them as first-class
    variables to the augmentor.
    """
    out = tmp_path / "stations_out.parquet"
    ec.enrich(
        stations_in,
        out,
        seifa_cache_dir=seifa_cache_dir,
        seifa_loader=seifa_loader,
        pipeline_factory=_make_stub_factory(lookup),
    )
    df = pd.read_parquet(out)
    derived = (
        "sa2_pct_drive_to_work",
        "sa2_motor_vehicles_per_dwelling",
        "sa2_pct_renters",
        "sa2_pct_employed_full_time",
        "sa2_pct_aged_65_plus",
        "sa2_pct_one_parent_family",
    )
    for col in derived:
        assert col in df.columns, f"missing column {col}"
        assert df[col].notna().all(), f"{col} should be populated, got {df[col].tolist()}"


def test_pipeline_receives_preset_variables(
    tmp_path: Path,
    stations_in: Path,
    seifa_cache_dir: Path,
    seifa_loader,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    """The DIRECT_VARIABLES dict passes 6 PRESET refs to the augmentor."""
    preset_keys = [
        k for k, v in ec.DIRECT_VARIABLES.items() if str(v).startswith("PRESET.")
    ]
    assert sorted(preset_keys) == sorted([
        "pct_drive_to_work",
        "motor_vehicles_per_dwelling",
        "pct_renters",
        "pct_employed_full_time",
        "pct_aged_65_plus",
        "pct_one_parent_family",
    ])
    # Each PRESET ref is well-formed (PRESET.<id> matching one we know upstream ships).
    for k, v in ec.DIRECT_VARIABLES.items():
        if v.startswith("PRESET."):
            preset_id = v.split(".", 1)[1]
            assert preset_id == k, (
                f"variable key {k!r} should match its PRESET id {preset_id!r}"
            )


def test_idempotent_skip_when_already_enriched(
    tmp_path: Path,
    seifa_cache_dir: Path,
    seifa_loader,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    """Re-running on a fully-enriched file should not call the augmentor."""
    df = pd.DataFrame(
        [
            {
                "station_id": "s1",
                "name": "BP Mascot",
                "lat": -33.93,
                "lon": 151.20,
                "sa2_code": "117011635",
                "sa2_name": "Mascot",
                "sa2_total_population": 21573,
                "sa2_median_age": 30,
                "sa2_median_household_income_weekly": 1900,
            }
        ]
    )
    p = tmp_path / "stations.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)

    stub = _StubPipeline(lookup)
    ec.enrich(
        p, p,
        seifa_cache_dir=seifa_cache_dir,
        seifa_loader=seifa_loader,
        pipeline_factory=lambda: stub,
    )

    assert stub.calls == []  # augmentor not invoked


def test_force_re_enriches_every_row(
    tmp_path: Path,
    seifa_cache_dir: Path,
    seifa_loader,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    df = pd.DataFrame(
        [
            {
                "station_id": "s1",
                "name": "BP Mascot",
                "lat": -33.93,
                "lon": 151.20,
                "sa2_code": "STALE",
                "sa2_total_population": 1,
            }
        ]
    )
    p = tmp_path / "stations.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)

    stub = _StubPipeline(lookup)
    ec.enrich(
        p, p,
        seifa_cache_dir=seifa_cache_dir,
        seifa_loader=seifa_loader,
        pipeline_factory=lambda: stub,
        force=True,
    )

    out = pd.read_parquet(p)
    assert out["sa2_code"].iloc[0] == "117011635"
    assert int(out["sa2_total_population"].iloc[0]) == 21573


def test_partial_enrichment_only_processes_unseen_rows(
    tmp_path: Path,
    seifa_cache_dir: Path,
    seifa_loader,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    """One row already has sa2_code → only the other gets sent to the augmentor."""
    df = pd.DataFrame(
        [
            {
                "station_id": "s1",
                "name": "BP Mascot",
                "lat": -33.93,
                "lon": 151.20,
                "sa2_code": "117011635",
                "sa2_total_population": 21573,
            },
            {
                "station_id": "s2",
                "name": "EG Ampol Newport",
                "lat": -33.65,
                "lon": 151.32,
            },
        ]
    )
    p = tmp_path / "stations.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)

    stub = _StubPipeline(lookup)
    ec.enrich(
        p, p,
        seifa_cache_dir=seifa_cache_dir,
        seifa_loader=seifa_loader,
        pipeline_factory=lambda: stub,
    )

    # The augmentor only saw s2 — even after the multi-pass split for
    # the upstream GCP collision workaround, every pass should be
    # restricted to the un-enriched row.
    assert len(stub.calls) >= 1
    for call in stub.calls:
        assert list(call["station_id"]) == ["s2"]


def test_missing_lat_lon_raises(
    tmp_path: Path, seifa_cache_dir: Path, seifa_loader
) -> None:
    df = pd.DataFrame([{"station_id": "s1", "name": "X"}])
    p = tmp_path / "stations.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(RuntimeError, match="lat/lon"):
        ec.enrich(
            p, p,
            seifa_cache_dir=seifa_cache_dir,
            seifa_loader=seifa_loader,
            pipeline_factory=lambda: _StubPipeline({}),
        )


def test_seifa_loader_failure_logs_warning_and_keeps_irsd_null(
    tmp_path: Path,
    stations_in: Path,
    seifa_cache_dir: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the augmentor's SEIFA fetch raises (e.g. ABS down), degrade gracefully."""
    out = tmp_path / "stations_out.parquet"

    def failing_loader() -> pd.DataFrame:
        raise RuntimeError("ABS unreachable")

    with caplog.at_level("WARNING", logger="fuel_pred.build.enrich_census"):
        ec.enrich(
            stations_in,
            out,
            seifa_cache_dir=seifa_cache_dir,
            seifa_loader=failing_loader,
            pipeline_factory=_make_stub_factory(lookup),
        )

    df = pd.read_parquet(out)
    assert df["sa2_seifa_irsd_score"].isna().all()
    assert any("SEIFA fetch via augmentor failed" in rec.message for rec in caplog.records)


def test_acceptance_warns_below_95_percent(
    tmp_path: Path,
    seifa_cache_dir: Path,
    seifa_loader,
    lookup: dict[tuple[float, float], dict[str, object]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Spec gate: log a clear warning if any required column < 95% coverage."""
    df = pd.DataFrame(
        [
            {"station_id": "s1", "name": "X", "lat": -33.93, "lon": 151.20},
            # Lat/lon points the stub doesn't recognise → enrichment fails.
            {"station_id": "s2", "name": "Y", "lat": 0.0, "lon": 0.0},
            {"station_id": "s3", "name": "Z", "lat": 0.0, "lon": 0.0},
        ]
    )
    p = tmp_path / "stations.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)

    with caplog.at_level("WARNING", logger="fuel_pred.build.enrich_census"):
        ec.enrich(
            p, p,
            seifa_cache_dir=seifa_cache_dir,
            seifa_loader=seifa_loader,
            pipeline_factory=_make_stub_factory(lookup),
        )

    # 1 of 3 enriched = 33%; well below 95%.
    assert any("threshold" in rec.message and "not met" in rec.message for rec in caplog.records)


def test_split_for_gcp_collision_passes_through_when_no_collision() -> None:
    """No collision (e.g. only direct vars or only PRESETs) → single group."""
    only_direct = {"pop": "G01.Tot_P_P", "age": "G02.Median_age_persons"}
    assert ec._split_for_gcp_collision(only_direct) == [only_direct]

    only_presets = {
        "rent": "PRESET.pct_renters",
        "drive": "PRESET.pct_drive_to_work",
    }
    assert ec._split_for_gcp_collision(only_presets) == [only_presets]


def test_split_for_gcp_collision_isolates_colliding_direct_var() -> None:
    """Direct G01.Tot_P_P + PRESET.pct_aged_65_plus splits into 2 groups.

    Workaround for upstream bug — PRESET.pct_aged_65_plus uses G01.Tot_P_P
    as its denominator, and the augmentor's GCP dispatch crashes when both
    are requested in one call. See module docstring (UPSTREAM_GCP_COLLISION).
    """
    variables = {
        "pop": "G01.Tot_P_P",
        "age": "G02.Median_age_persons",
        "aged": "PRESET.pct_aged_65_plus",
        "rent": "PRESET.pct_renters",
    }
    groups = ec._split_for_gcp_collision(variables)
    assert len(groups) == 2
    # All vars accounted for, exactly once.
    seen = {k for g in groups for k in g}
    assert seen == set(variables)
    # The colliding direct ref ('pop') is isolated from the colliding PRESET ('aged').
    pop_group = next(g for g in groups if "pop" in g)
    aged_group = next(g for g in groups if "aged" in g)
    assert pop_group is not aged_group
    # Non-colliding entries (age, rent) ride along with the PRESETs;
    # only the colliding direct ref is split out into its own pass.
    assert "aged" in aged_group and "rent" in aged_group and "age" in aged_group
    assert pop_group == {"pop": "G01.Tot_P_P"}


def test_in_place_overwrite_is_atomic(
    tmp_path: Path,
    stations_in: Path,
    seifa_cache_dir: Path,
    seifa_loader,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    ec.enrich(
        stations_in,
        stations_in,
        seifa_cache_dir=seifa_cache_dir,
        seifa_loader=seifa_loader,
        pipeline_factory=_make_stub_factory(lookup),
    )
    df = pd.read_parquet(stations_in)
    assert "sa2_code" in df.columns
    assert not (stations_in.parent / (stations_in.name + ".tmp")).exists()
