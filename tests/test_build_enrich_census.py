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
def seifa_path(tmp_path: Path) -> Path:
    """A tiny SEIFA parquet — only the irsd columns needed by the join."""
    df = pd.DataFrame(
        [
            {"sa2_code": "117011635", "irsd_score": 1098, "sa2_name": "Mascot"},
            {"sa2_code": "122021422", "irsd_score": 1102, "sa2_name": "Newport"},
            {"sa2_code": "999999999", "irsd_score": 999, "sa2_name": "Unused"},
        ]
    )
    p = tmp_path / "seifa.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    return p


@pytest.fixture
def lookup() -> dict[tuple[float, float], dict[str, object]]:
    return {
        (-33.93, 151.20): {
            "sa2_code": "117011635",
            "sa2_name": "Mascot",
            "sa2_total_population": 21573,
            "sa2_median_age": 30,
            "sa2_median_household_income_weekly": 1900,
        },
        (-33.65, 151.32): {
            "sa2_code": "122021422",
            "sa2_name": "Newport - Bilgola",
            "sa2_total_population": 13681,
            "sa2_median_age": 46,
            "sa2_median_household_income_weekly": 2400,
        },
    }


def test_writes_full_schema(
    tmp_path: Path,
    stations_in: Path,
    seifa_path: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    out = tmp_path / "stations_out.parquet"
    ec.enrich(
        stations_in,
        out,
        seifa_path=seifa_path,
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
    seifa_path: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    out = tmp_path / "stations_out.parquet"
    ec.enrich(
        stations_in,
        out,
        seifa_path=seifa_path,
        pipeline_factory=_make_stub_factory(lookup),
    )
    df = pd.read_parquet(out)
    by_id = {row["station_id"]: row for _, row in df.iterrows()}
    assert int(by_id["s1"]["sa2_seifa_irsd_score"]) == 1098
    assert int(by_id["s2"]["sa2_seifa_irsd_score"]) == 1102


def test_six_derived_columns_are_null_stubs(
    tmp_path: Path,
    stations_in: Path,
    seifa_path: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    """Per spec §7.7.1, the 6 derived percentages exist but are null in v1."""
    out = tmp_path / "stations_out.parquet"
    ec.enrich(
        stations_in,
        out,
        seifa_path=seifa_path,
        pipeline_factory=_make_stub_factory(lookup),
    )
    df = pd.read_parquet(out)
    for col in ec.DEFERRED_DERIVED_COLUMNS:
        assert col in df.columns, f"missing stub column {col}"
        assert df[col].isna().all(), f"{col} should be null in v1, got {df[col].tolist()}"


def test_idempotent_skip_when_already_enriched(
    tmp_path: Path, seifa_path: Path, lookup: dict[tuple[float, float], dict[str, object]]
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
    ec.enrich(p, p, seifa_path=seifa_path, pipeline_factory=lambda: stub)

    assert stub.calls == []  # augmentor not invoked


def test_force_re_enriches_every_row(
    tmp_path: Path, seifa_path: Path, lookup: dict[tuple[float, float], dict[str, object]]
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
    ec.enrich(p, p, seifa_path=seifa_path, pipeline_factory=lambda: stub, force=True)

    out = pd.read_parquet(p)
    assert out["sa2_code"].iloc[0] == "117011635"
    assert int(out["sa2_total_population"].iloc[0]) == 21573


def test_partial_enrichment_only_processes_unseen_rows(
    tmp_path: Path, seifa_path: Path, lookup: dict[tuple[float, float], dict[str, object]]
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
    ec.enrich(p, p, seifa_path=seifa_path, pipeline_factory=lambda: stub)

    # The augmentor only saw s2.
    assert len(stub.calls) == 1
    assert list(stub.calls[0]["station_id"]) == ["s2"]


def test_missing_lat_lon_raises(tmp_path: Path, seifa_path: Path) -> None:
    df = pd.DataFrame([{"station_id": "s1", "name": "X"}])
    p = tmp_path / "stations.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(RuntimeError, match="lat/lon"):
        ec.enrich(p, p, seifa_path=seifa_path, pipeline_factory=lambda: _StubPipeline({}))


def test_seifa_missing_logs_warning_and_keeps_irsd_null(
    tmp_path: Path,
    stations_in: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    out = tmp_path / "stations_out.parquet"
    seifa_missing = tmp_path / "seifa_does_not_exist.parquet"

    with caplog.at_level("WARNING", logger="fuel_pred.build.enrich_census"):
        ec.enrich(
            stations_in,
            out,
            seifa_path=seifa_missing,
            pipeline_factory=_make_stub_factory(lookup),
        )

    df = pd.read_parquet(out)
    assert df["sa2_seifa_irsd_score"].isna().all()
    assert any("SEIFA parquet missing" in rec.message for rec in caplog.records)


def test_acceptance_warns_below_95_percent(
    tmp_path: Path, seifa_path: Path, lookup: dict[tuple[float, float], dict[str, object]],
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
        ec.enrich(p, p, seifa_path=seifa_path, pipeline_factory=_make_stub_factory(lookup))

    # 1 of 3 enriched = 33%; well below 95%.
    assert any("threshold" in rec.message and "not met" in rec.message for rec in caplog.records)


def test_in_place_overwrite_is_atomic(
    tmp_path: Path,
    stations_in: Path,
    seifa_path: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    ec.enrich(
        stations_in,
        stations_in,
        seifa_path=seifa_path,
        pipeline_factory=_make_stub_factory(lookup),
    )
    df = pd.read_parquet(stations_in)
    assert "sa2_code" in df.columns
    assert not (stations_in.parent / (stations_in.name + ".tmp")).exists()
