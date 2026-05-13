"""Hermetic tests for build.enrich_census.

The augmentor's `Pipeline` is replaced with a stub that returns a
deterministic enriched frame — no S3, no boundaries download, no
DSS / ERP / ABS_PIA fetches.

Augmentor v1.5+ unifies SEIFA / ERP / DSS / ABS_PIA dispatch through
``Pipeline.augment(...)``; the previous bespoke ``SeifaDataSource``
code path (and its ``seifa_loader`` test seam) is gone, so SEIFA
columns now arrive on the same enriched frame as everything else.
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
    """Minimal stand-in for `census_augment.Pipeline.create(...)`.

    Builds an output frame with every column listed in
    ``ec.ENRICHED_COLUMNS`` so the test surface tracks the full v1.5
    schema (28 columns at time of writing). Per-row values come from
    ``sa2_lookup`` keyed on (rounded lat, rounded lon).
    """

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
        for col in ec.ENRICHED_COLUMNS:
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
def lookup() -> dict[tuple[float, float], dict[str, object]]:
    """A single per-station enrichment payload covering every column the
    real augmentor would produce post v1.5.
    """
    return {
        (-33.93, 151.20): {
            "sa2_code": "117011635",
            "sa2_name": "Mascot",
            # GCP direct
            "sa2_total_population": 21573,
            "sa2_median_age": 30,
            "sa2_median_household_income_weekly": 1900,
            # PRESETs
            "sa2_pct_drive_to_work": 38.5,
            "sa2_motor_vehicles_per_dwelling": 0.62,
            "sa2_pct_renters": 65.85,
            "sa2_pct_employed_full_time": 63.0,
            "sa2_pct_aged_65_plus": 14.82,
            "sa2_pct_one_parent_family": 18.0,
            # SEIFA
            "sa2_seifa_irsd_score": 1098.0,
            "sa2_seifa_irsad_score": 1102.0,
            "sa2_seifa_ier_score": 1085.0,
            "sa2_seifa_ieo_score": 1110.0,
            # ERP
            "sa2_erp_population_density_per_km2": 6900.0,
            "sa2_erp_population_0_14": 3500,
            "sa2_erp_population_15_64": 16100,
            "sa2_erp_population_65_plus": 1973,
            "sa2_erp_median_age": 30.5,
            # ABS_PIA
            "sa2_pia_gini_coefficient": 0.42,
            # DSS
            "sa2_dss_age_pension_recipients": 850,
            "sa2_dss_jobseeker_payment_recipients": 410,
            "sa2_dss_disability_support_pension_recipients": 130,
            "sa2_dss_parenting_payment_single_recipients": 90,
            "sa2_dss_parenting_payment_partnered_recipients": 30,
            "sa2_dss_carer_payment_recipients": 75,
            "sa2_dss_youth_allowance_other_recipients": 60,
            "sa2_dss_youth_allowance_student_recipients": 200,
            "sa2_dss_commonwealth_rent_assistance_recipients": 4200,
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
            "sa2_seifa_irsd_score": 1102.0,
            "sa2_seifa_irsad_score": 1140.0,
            "sa2_seifa_ier_score": 1095.0,
            "sa2_seifa_ieo_score": 1130.0,
            "sa2_erp_population_density_per_km2": 1100.0,
            "sa2_erp_population_0_14": 2400,
            "sa2_erp_population_15_64": 8100,
            "sa2_erp_population_65_plus": 3181,
            "sa2_erp_median_age": 46.2,
            "sa2_pia_gini_coefficient": 0.48,
            "sa2_dss_age_pension_recipients": 1250,
            "sa2_dss_jobseeker_payment_recipients": 180,
            "sa2_dss_disability_support_pension_recipients": 110,
            "sa2_dss_parenting_payment_single_recipients": 50,
            "sa2_dss_parenting_payment_partnered_recipients": 15,
            "sa2_dss_carer_payment_recipients": 90,
            "sa2_dss_youth_allowance_other_recipients": 25,
            "sa2_dss_youth_allowance_student_recipients": 120,
            "sa2_dss_commonwealth_rent_assistance_recipients": 1500,
        },
    }


def test_writes_full_schema(
    tmp_path: Path,
    stations_in: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    out = tmp_path / "stations_out.parquet"
    ec.enrich(
        stations_in,
        out,
        pipeline_factory=_make_stub_factory(lookup),
    )

    df = pd.read_parquet(out)
    for col in ec.ENRICHED_COLUMNS:
        assert col in df.columns, f"missing column {col}"

    # Both stations enriched on the dense (GCP / SEIFA) columns.
    assert df["sa2_code"].notna().all()
    assert df["sa2_total_population"].notna().all()
    assert df["sa2_seifa_irsd_score"].notna().all()
    assert df["sa2_seifa_irsad_score"].notna().all()


def test_seifa_indexes_populate_via_unified_dispatch(
    tmp_path: Path,
    stations_in: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    """v1.5 routes ``SEIFA.<field>`` through the same Pipeline.augment call
    as everything else. All four SEIFA scores land on the output frame.
    """
    out = tmp_path / "stations_out.parquet"
    ec.enrich(
        stations_in,
        out,
        pipeline_factory=_make_stub_factory(lookup),
    )
    df = pd.read_parquet(out)
    by_id = {row["station_id"]: row for _, row in df.iterrows()}
    assert int(by_id["s1"]["sa2_seifa_irsd_score"]) == 1098
    assert int(by_id["s1"]["sa2_seifa_irsad_score"]) == 1102
    assert int(by_id["s1"]["sa2_seifa_ier_score"]) == 1085
    assert int(by_id["s1"]["sa2_seifa_ieo_score"]) == 1110
    assert int(by_id["s2"]["sa2_seifa_irsd_score"]) == 1102


def test_six_preset_derivations_populate(
    tmp_path: Path,
    stations_in: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    """The 6 PRESET-derived percentages populate per row (augmentor v1.4.2+)."""
    out = tmp_path / "stations_out.parquet"
    ec.enrich(
        stations_in,
        out,
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


def test_new_dataset_columns_populate(
    tmp_path: Path,
    stations_in: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    """ERP / ABS_PIA / DSS columns from the v1.5 surface land on the output."""
    out = tmp_path / "stations_out.parquet"
    ec.enrich(
        stations_in,
        out,
        pipeline_factory=_make_stub_factory(lookup),
    )
    df = pd.read_parquet(out)
    for prefix in ("sa2_erp_", "sa2_pia_", "sa2_dss_"):
        cols = [c for c in df.columns if c.startswith(prefix)]
        assert cols, f"no columns with prefix {prefix!r} on output"
        for col in cols:
            assert df[col].notna().all(), (
                f"{col} should be populated by the stub, got {df[col].tolist()}"
            )


def test_pipeline_receives_preset_variables(
    tmp_path: Path,
    stations_in: Path,
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


def test_pipeline_includes_all_v15_namespaces() -> None:
    """DIRECT_VARIABLES exercises every v1.5 dispatch namespace we expect.

    The augmentor v1.5 surface registers G (GCP), PRESET, SEIFA, ERP,
    ABS_PIA, and DSS. Phase 3 of this project pulls from all six.
    """
    namespaces = {v.split(".", 1)[0] for v in ec.DIRECT_VARIABLES.values()}
    expected = {"G01", "G02", "PRESET", "SEIFA", "ERP", "ABS_PIA", "DSS"}
    missing = expected - namespaces
    assert not missing, f"DIRECT_VARIABLES missing namespaces: {missing}"


def test_idempotent_skip_when_already_enriched(
    tmp_path: Path,
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
        pipeline_factory=lambda: stub,
    )

    assert stub.calls == []  # augmentor not invoked


def test_force_re_enriches_every_row(
    tmp_path: Path,
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
        pipeline_factory=lambda: stub,
        force=True,
    )

    out = pd.read_parquet(p)
    assert out["sa2_code"].iloc[0] == "117011635"
    assert int(out["sa2_total_population"].iloc[0]) == 21573


def test_partial_enrichment_only_processes_unseen_rows(
    tmp_path: Path,
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
        pipeline_factory=lambda: stub,
    )

    # The augmentor only saw s2 — even after the multi-pass split for
    # the upstream GCP collision workaround, every pass should be
    # restricted to the un-enriched row.
    assert len(stub.calls) >= 1
    for call in stub.calls:
        assert list(call["station_id"]) == ["s2"]


def test_missing_lat_lon_raises(tmp_path: Path) -> None:
    df = pd.DataFrame([{"station_id": "s1", "name": "X"}])
    p = tmp_path / "stations.parquet"
    df.to_parquet(p, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(RuntimeError, match="lat/lon"):
        ec.enrich(
            p, p,
            pipeline_factory=lambda: _StubPipeline({}),
        )


def test_acceptance_warns_below_95_percent_on_strict_columns(
    tmp_path: Path,
    lookup: dict[tuple[float, float], dict[str, object]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Spec gate: log a clear warning if any *strict* column < 95% coverage.

    ERP / ABS_PIA / DSS columns are NOT in the strict set (they have
    legitimate per-SA2 nulls), so partial enrichment of those alone
    should not trigger a warning.
    """
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
            pipeline_factory=_make_stub_factory(lookup),
        )

    # 1 of 3 enriched = 33%; well below 95% on the strict columns.
    def _is_strict_breach(msg: str) -> bool:
        return (
            "threshold" in msg
            and "not met" in msg
            and "sa2_total_population" in msg
        )

    assert any(_is_strict_breach(rec.message) for rec in caplog.records)


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
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    ec.enrich(
        stations_in,
        stations_in,
        pipeline_factory=_make_stub_factory(lookup),
    )
    df = pd.read_parquet(stations_in)
    assert "sa2_code" in df.columns
    assert not (stations_in.parent / (stations_in.name + ".tmp")).exists()
