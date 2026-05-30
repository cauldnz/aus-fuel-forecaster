"""Hermetic tests for build.enrich_panel_temporal.

Same stub-pipeline pattern as test_build_enrich_census: replace the
augmentor's `Pipeline.augment` with a stub that returns a deterministic
DataFrame. No network, no boundaries download.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from fuel_pred.build import enrich_panel_temporal as ept


@dataclass
class _StubResult:
    df: pd.DataFrame
    releases_used: dict[str, list[str]]


class _StubTemporalPipeline:
    """Returns a frame keyed by (lat, lon, date) → demographic payload.

    Mimics the augmentor's temporal output: input frame + appended
    ``sa2_<col>`` value columns + bookkeeping columns (``sa2_code``,
    ``<dataset>_release``, etc.). The bookkeeping columns are present so
    the module's projection step is exercised end-to-end.
    """

    def __init__(self, payload_by_key: dict[tuple[float, float, dt.date], dict[str, object]]):
        self.payload_by_key = payload_by_key
        self.calls: list[pd.DataFrame] = []

    def augment(self, df: pd.DataFrame) -> _StubResult:
        self.calls.append(df.copy())
        out = df.copy()
        # All ``sa2_*`` value columns for the requested TEMPORAL_VARIABLES
        for key in ept.TEMPORAL_VARIABLES:
            out[f"sa2_{key}"] = pd.NA
        # Bookkeeping columns the augmentor emits in temporal mode
        out["sa2_code"] = pd.NA
        out["sa2_name"] = pd.NA
        out["sa2_code_edition"] = 3
        out["seifa_release"] = "2021"
        out["erp_by_sa2_release"] = "2023"
        out["dss_payments_release"] = "2024-Q4"

        for idx, row in df.iterrows():
            key = (round(row["lat"], 4), round(row["lon"], 4), row["date"])
            data = self.payload_by_key.get(key)
            if data is not None:
                for col, val in data.items():
                    out.at[idx, col] = val
        return _StubResult(df=out, releases_used={"seifa": ["2016", "2021"]})


def _make_factory(payload: dict[tuple[float, float, dt.date], dict[str, object]]):
    def factory(variables: dict[str, str] | None = None) -> _StubTemporalPipeline:
        return _StubTemporalPipeline(payload)

    return factory


# --------------------- fixtures ---------------------


@pytest.fixture
def panel_in(tmp_path: Path) -> Path:
    """Panel: 2 stations × 2 fuels × 2 dates = 8 rows; unique (sid, date) = 4."""
    rows = []
    for sid in ("s1", "s2"):
        for fuel in ("U91", "Diesel"):
            for d in (dt.date(2017, 6, 15), dt.date(2023, 6, 15)):
                rows.append({"station_id": sid, "fuel_code": fuel, "date": d})
    p = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    return p


@pytest.fixture
def stations_in(tmp_path: Path) -> Path:
    p = tmp_path / "stations.parquet"
    pd.DataFrame(
        [
            {"station_id": "s1", "lat": -33.93, "lon": 151.20},
            {"station_id": "s2", "lat": -33.65, "lon": 151.32},
        ]
    ).to_parquet(p, engine="pyarrow", compression="zstd", index=False)
    return p


@pytest.fixture
def payload() -> dict[tuple[float, float, dt.date], dict[str, object]]:
    """Two stations × two dates of fully-populated SA2 temporal columns.

    DSS columns are intentionally absent — they sit in the cross-sectional
    pass today (see config.AUGMENTOR_VARIABLES_TEMPORAL note about the
    upstream DSS 2022-Q4 parser bug).
    """

    def make(seifa_irsd: float, erp: int) -> dict[str, object]:
        return {
            "sa2_seifa_irsd_score": seifa_irsd,
            "sa2_seifa_irsad_score": seifa_irsd + 5,
            "sa2_seifa_ier_score": seifa_irsd - 5,
            "sa2_seifa_ieo_score": seifa_irsd + 10,
            "sa2_erp_population_total": erp,
        }

    return {
        # s1 in 2017 + 2023, then s2
        (-33.93, 151.20, dt.date(2017, 6, 15)): make(1090, 21000),
        (-33.93, 151.20, dt.date(2023, 6, 15)): make(1095, 21500),
        (-33.65, 151.32, dt.date(2017, 6, 15)): make(1100, 13700),
        (-33.65, 151.32, dt.date(2023, 6, 15)): make(1115, 13900),
    }


# --------------------- behaviour ---------------------


def test_deduplicates_panel_to_unique_station_date(
    tmp_path: Path,
    panel_in: Path,
    stations_in: Path,
    payload: dict,
) -> None:
    """8 panel rows → 4 unique (station_id, date) sent to augmentor."""
    out_path = tmp_path / "panel_sa2_temporal.parquet"
    factory = _make_factory(payload)
    ept.enrich(panel_in, stations_in, out_path, pipeline_factory=factory)

    # 8 input panel rows (2 stations × 2 fuels × 2 dates) dedupe to
    # 4 unique (station_id, date) tuples — fuel is irrelevant to SA2.
    assert out_path.exists()
    out = pd.read_parquet(out_path)
    assert len(out) == 4


def test_output_schema_matches_expected_columns(
    tmp_path: Path,
    panel_in: Path,
    stations_in: Path,
    payload: dict,
) -> None:
    """Output frame has exactly (station_id, date) + sa2_* value cols."""
    out_path = tmp_path / "panel_sa2_temporal.parquet"
    ept.enrich(panel_in, stations_in, out_path, pipeline_factory=_make_factory(payload))
    out = pd.read_parquet(out_path)
    assert list(out.columns) == list(ept.OUTPUT_COLUMNS)
    # Bookkeeping cols dropped (sa2_code, *_release, etc.)
    for stripped in ("sa2_code", "seifa_release", "erp_by_sa2_release"):
        assert stripped not in out.columns


def test_values_come_through_correctly(
    tmp_path: Path,
    panel_in: Path,
    stations_in: Path,
    payload: dict,
) -> None:
    """Stub's per-key payload reaches the output by (station, date) join."""
    out_path = tmp_path / "panel_sa2_temporal.parquet"
    ept.enrich(panel_in, stations_in, out_path, pipeline_factory=_make_factory(payload))
    out = pd.read_parquet(out_path)
    s1_2017 = out[(out["station_id"] == "s1") & (out["date"] == dt.date(2017, 6, 15))]
    assert len(s1_2017) == 1
    assert float(s1_2017["sa2_seifa_irsd_score"].iloc[0]) == 1090.0
    s2_2023 = out[(out["station_id"] == "s2") & (out["date"] == dt.date(2023, 6, 15))]
    assert float(s2_2023["sa2_seifa_irsd_score"].iloc[0]) == 1115.0


def test_drops_stations_without_latlon(
    tmp_path: Path,
    panel_in: Path,
    payload: dict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stations with null lat/lon are dropped + warning logged."""
    stations_in = tmp_path / "stations.parquet"
    pd.DataFrame(
        [
            {"station_id": "s1", "lat": -33.93, "lon": 151.20},
            {"station_id": "s2", "lat": None, "lon": None},  # missing lat/lon
        ]
    ).to_parquet(stations_in, engine="pyarrow", compression="zstd", index=False)
    out_path = tmp_path / "panel_sa2_temporal.parquet"
    with caplog.at_level(logging.WARNING):
        ept.enrich(panel_in, stations_in, out_path, pipeline_factory=_make_factory(payload))
    out = pd.read_parquet(out_path)
    # Only s1 should land
    assert set(out["station_id"]) == {"s1"}
    assert any("missing lat/lon" in rec.message for rec in caplog.records)


def test_raises_when_no_usable_rows(
    tmp_path: Path,
    panel_in: Path,
    payload: dict,
) -> None:
    """If every station lacks lat/lon, refuse to write an empty output."""
    stations_in = tmp_path / "stations.parquet"
    pd.DataFrame(
        [
            {"station_id": "s1", "lat": None, "lon": None},
            {"station_id": "s2", "lat": None, "lon": None},
        ]
    ).to_parquet(stations_in, engine="pyarrow", compression="zstd", index=False)
    out_path = tmp_path / "panel_sa2_temporal.parquet"
    with pytest.raises(RuntimeError, match="no usable rows"):
        ept.enrich(panel_in, stations_in, out_path, pipeline_factory=_make_factory(payload))


def test_temporal_variables_subset_of_augmentor_config() -> None:
    """The module's TEMPORAL_VARIABLES must mirror config.AUGMENTOR_VARIABLES_TEMPORAL."""
    from fuel_pred import config as cfg

    assert dict(cfg.AUGMENTOR_VARIABLES_TEMPORAL) == ept.TEMPORAL_VARIABLES


def test_temporal_variables_do_not_overlap_cross_sectional() -> None:
    """No key appears in both passes (spec §7.7.2 split is exhaustive)."""
    from fuel_pred import config as cfg

    overlap = set(cfg.AUGMENTOR_VARIABLES_TEMPORAL) & set(cfg.AUGMENTOR_VARIABLES_CROSS_SECTIONAL)
    assert overlap == set(), f"variables in both passes: {sorted(overlap)}"


def test_temporal_variables_include_seifa_and_erp_total() -> None:
    """Headline variables for PR B's experiment must be in the temporal set.

    DSS is intentionally NOT in the temporal set today — the augmentor's
    DSS XLSX parser fails on the 2022-Q4 release (oldest available),
    filed upstream. DSS stays cross-sectional in PR B; this assertion
    locks the door against accidentally re-adding it before the upstream
    parser issue lands.
    """
    keys = set(ept.TEMPORAL_VARIABLES)
    # SEIFA: all 4 scores
    assert {"seifa_irsd_score", "seifa_irsad_score", "seifa_ier_score", "seifa_ieo_score"} <= keys
    # ERP population_total (not the age/sex cols)
    assert "erp_population_total" in keys
    assert "erp_population_65_plus" not in keys
    # DSS held back pending upstream parser fix.
    assert not any(k.startswith("dss_") for k in keys), (
        "DSS variables should not be temporal yet — upstream parser issue "
        "on 2022-Q4. Move once it lands."
    )
