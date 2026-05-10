"""Hermetic tests for build.panel_grid.

Forward-fill semantics are silent failure modes — the spec ships
``max_forward_fill_days=7``, so tests pin the boundary behaviour at
the 7-day cliff and the immediate-after-cliff null transition.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from fuel_pred.build import panel_grid as pg


def _stations(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("first_seen", "last_seen"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date
    return df


def _fuel_daily(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ----------------------------- shape -----------------------------


def test_panel_has_expected_schema() -> None:
    stations = _stations(
        [{"station_id": "s1", "first_seen": "2024-01-01", "last_seen": "2024-01-03"}]
    )
    fuel_daily = _fuel_daily(
        [
            {
                "station_id": "s1",
                "fuel_code": "U91",
                "date": "2024-01-01",
                "price_mean": 200.0,
                "price_min": 199.0,
                "price_max": 201.0,
                "n_obs": 3,
            }
        ]
    )

    panel = pg.build(stations, fuel_daily, span_start="2024-01-01")
    assert list(panel.columns) == list(pg.OUTPUT_COLUMNS)


def test_grid_is_dense_per_station_fuel_date() -> None:
    """3-day window x 2 fuels x 1 station = 6 rows, regardless of how
    many price observations existed."""
    stations = _stations(
        [{"station_id": "s1", "first_seen": "2024-01-01", "last_seen": "2024-01-03"}]
    )
    fuel_daily = _fuel_daily([])  # no price observations at all

    panel = pg.build(stations, fuel_daily, span_start="2024-01-01")
    assert len(panel) == 6
    grouped = panel.groupby(["station_id", "fuel_code"]).size().to_dict()
    assert grouped == {("s1", "U91"): 3, ("s1", "DL"): 3}


def test_both_fuels_present_even_when_only_one_has_observations() -> None:
    """U91-only data must still produce the DL rows (with nulls) — the
    feature builder needs both fuels for the cross-fuel block."""
    stations = _stations(
        [{"station_id": "s1", "first_seen": "2024-01-01", "last_seen": "2024-01-02"}]
    )
    fuel_daily = _fuel_daily(
        [
            {
                "station_id": "s1",
                "fuel_code": "U91",
                "date": "2024-01-01",
                "price_mean": 200.0,
                "price_min": 199.0,
                "price_max": 201.0,
                "n_obs": 3,
            }
        ]
    )
    panel = pg.build(stations, fuel_daily, span_start="2024-01-01")
    fuels = set(panel["fuel_code"].unique())
    assert fuels == {"U91", "DL"}
    dl_rows = panel[panel["fuel_code"] == "DL"]
    assert dl_rows["price_mean"].isna().all()
    assert (dl_rows["n_obs"] == 0).all()


# ----------------------------- forward-fill semantics -----------------------------


def test_forward_fill_extends_through_gap_within_horizon() -> None:
    """Single observation on day 0; days 1..7 should carry that value forward."""
    dates_with_obs = ["2024-01-01"]
    full_dates = pd.date_range("2024-01-01", "2024-01-08").date.tolist()

    stations = _stations(
        [
            {
                "station_id": "s1",
                "first_seen": full_dates[0].isoformat(),
                "last_seen": full_dates[-1].isoformat(),
            }
        ]
    )
    fuel_daily = _fuel_daily(
        [
            {
                "station_id": "s1",
                "fuel_code": "U91",
                "date": d,
                "price_mean": 200.0,
                "price_min": 199.0,
                "price_max": 201.0,
                "n_obs": 3,
            }
            for d in dates_with_obs
        ]
    )

    panel = pg.build(
        stations, fuel_daily, span_start=full_dates[0].isoformat(), max_forward_fill_days=7
    )
    u91 = panel[panel["fuel_code"] == "U91"].sort_values("date")
    # 8 days: day 0 = real, days 1..7 = forward-filled (within horizon).
    assert u91["price_mean"].notna().sum() == 8


def test_forward_fill_nulls_after_horizon_cliff() -> None:
    """Single observation on day 0; day 8 onwards (>7 day gap) must be null."""
    full_dates = pd.date_range("2024-01-01", "2024-01-15").date.tolist()
    stations = _stations(
        [
            {
                "station_id": "s1",
                "first_seen": full_dates[0].isoformat(),
                "last_seen": full_dates[-1].isoformat(),
            }
        ]
    )
    fuel_daily = _fuel_daily(
        [
            {
                "station_id": "s1",
                "fuel_code": "U91",
                "date": "2024-01-01",
                "price_mean": 200.0,
                "price_min": 199.0,
                "price_max": 201.0,
                "n_obs": 3,
            }
        ]
    )

    panel = pg.build(
        stations, fuel_daily, span_start=full_dates[0].isoformat(), max_forward_fill_days=7
    )
    u91 = panel[panel["fuel_code"] == "U91"].sort_values("date")
    # Days 1..7 carried forward; day 8 onwards nulled (gap > horizon).
    expected_filled_dates = pd.date_range("2024-01-01", "2024-01-08").date.tolist()
    filled_in_panel = set(u91.loc[u91["price_mean"].notna(), "date"])
    assert filled_in_panel == set(expected_filled_dates)


def test_forward_fill_resumes_after_a_real_observation() -> None:
    """A new real observation in the middle resets the forward-fill horizon."""
    full_dates = pd.date_range("2024-01-01", "2024-01-20").date.tolist()
    stations = _stations(
        [
            {
                "station_id": "s1",
                "first_seen": full_dates[0].isoformat(),
                "last_seen": full_dates[-1].isoformat(),
            }
        ]
    )
    # Observations on day 0 and day 10 — day 10's value resets the gap counter.
    fuel_daily = _fuel_daily(
        [
            {
                "station_id": "s1",
                "fuel_code": "U91",
                "date": "2024-01-01",
                "price_mean": 200.0,
                "price_min": 199.0,
                "price_max": 201.0,
                "n_obs": 3,
            },
            {
                "station_id": "s1",
                "fuel_code": "U91",
                "date": "2024-01-10",
                "price_mean": 210.0,
                "price_min": 209.0,
                "price_max": 211.0,
                "n_obs": 2,
            },
        ]
    )

    panel = pg.build(
        stations, fuel_daily, span_start=full_dates[0].isoformat(), max_forward_fill_days=7
    )
    u91 = panel[panel["fuel_code"] == "U91"].sort_values("date").reset_index(drop=True)
    # max_forward_fill_days=7 → value carries for 7 days *after* the
    # observation. Day 0 = real; days 1..7 = filled; day 8 = first stale.
    # 2024-01-01 + 7 days = 2024-01-08 (still filled, days_since=7).
    # 2024-01-01 + 8 days = 2024-01-09 (null, days_since=8).
    by_date = {row["date"].isoformat(): row["price_mean"] for _, row in u91.iterrows()}
    assert by_date["2024-01-01"] == 200.0  # real
    assert by_date["2024-01-08"] == 200.0  # 7th filled day (days_since=7)
    assert pd.isna(by_date["2024-01-09"])  # days_since=8, gap exceeded
    assert by_date["2024-01-10"] == 210.0  # real
    # Counter resets at day 10's real obs; carries 7 more days to 2024-01-17.
    assert by_date["2024-01-17"] == 210.0  # days_since=7 from 2024-01-10
    assert pd.isna(by_date["2024-01-18"])  # days_since=8 from 2024-01-10


def test_forward_fill_does_not_cross_station_boundary() -> None:
    """A run from station s1 must not bleed into s2's first day."""
    stations = _stations(
        [
            {"station_id": "s1", "first_seen": "2024-01-01", "last_seen": "2024-01-03"},
            {"station_id": "s2", "first_seen": "2024-01-01", "last_seen": "2024-01-03"},
        ]
    )
    fuel_daily = _fuel_daily(
        [
            {
                "station_id": "s1",
                "fuel_code": "U91",
                "date": "2024-01-01",
                "price_mean": 200.0,
                "price_min": 199.0,
                "price_max": 201.0,
                "n_obs": 3,
            }
            # No observations for s2 at all.
        ]
    )

    panel = pg.build(stations, fuel_daily, span_start="2024-01-01")
    s2_u91 = panel[(panel["station_id"] == "s2") & (panel["fuel_code"] == "U91")]
    assert s2_u91["price_mean"].isna().all()


def test_forward_fill_does_not_cross_fuel_boundary() -> None:
    """An s1 U91 value at day 0 must not fill into s1 DL at day 1."""
    stations = _stations(
        [{"station_id": "s1", "first_seen": "2024-01-01", "last_seen": "2024-01-02"}]
    )
    fuel_daily = _fuel_daily(
        [
            {
                "station_id": "s1",
                "fuel_code": "U91",
                "date": "2024-01-01",
                "price_mean": 200.0,
                "price_min": 199.0,
                "price_max": 201.0,
                "n_obs": 3,
            }
        ]
    )
    panel = pg.build(stations, fuel_daily, span_start="2024-01-01")
    dl = panel[panel["fuel_code"] == "DL"]
    assert dl["price_mean"].isna().all()


# ----------------------------- date-range edge cases -----------------------------


def test_station_first_seen_clamped_to_span_start() -> None:
    """A station with first_seen=2010-01-01 in a span starting 2024 must
    only get rows from 2024 onwards."""
    stations = _stations(
        [{"station_id": "s1", "first_seen": "2010-01-01", "last_seen": "2024-01-05"}]
    )
    fuel_daily = _fuel_daily([])
    panel = pg.build(stations, fuel_daily, span_start="2024-01-01")
    assert panel["date"].min() == dt.date(2024, 1, 1)


def test_span_end_clamps_last_seen() -> None:
    stations = _stations(
        [{"station_id": "s1", "first_seen": "2024-01-01", "last_seen": "2030-12-31"}]
    )
    fuel_daily = _fuel_daily([])
    panel = pg.build(
        stations, fuel_daily, span_start="2024-01-01", span_end="2024-01-05"
    )
    assert panel["date"].max() == dt.date(2024, 1, 5)


def test_station_with_no_dates_contributes_no_rows() -> None:
    """If first_seen > last_seen (degenerate), the station produces zero rows."""
    stations = _stations(
        [
            {"station_id": "s1", "first_seen": "2024-01-10", "last_seen": "2024-01-05"},
            {"station_id": "s2", "first_seen": "2024-01-01", "last_seen": "2024-01-03"},
        ]
    )
    fuel_daily = _fuel_daily([])
    panel = pg.build(stations, fuel_daily, span_start="2024-01-01")
    assert "s1" not in panel["station_id"].unique()
    assert "s2" in panel["station_id"].unique()


def test_n_obs_is_zero_on_inserted_rows() -> None:
    stations = _stations(
        [{"station_id": "s1", "first_seen": "2024-01-01", "last_seen": "2024-01-03"}]
    )
    fuel_daily = _fuel_daily(
        [
            {
                "station_id": "s1",
                "fuel_code": "U91",
                "date": "2024-01-01",
                "price_mean": 200.0,
                "price_min": 199.0,
                "price_max": 201.0,
                "n_obs": 5,
            }
        ]
    )
    panel = pg.build(stations, fuel_daily, span_start="2024-01-01")
    u91 = panel[panel["fuel_code"] == "U91"].sort_values("date").reset_index(drop=True)
    # n_obs = 5 on day 0, 0 on days 1 and 2 (inserted rows; forward-fill
    # affects only price columns).
    assert u91.iloc[0]["n_obs"] == 5
    assert u91.iloc[1]["n_obs"] == 0
    assert u91.iloc[2]["n_obs"] == 0


# ----------------------------- file IO -----------------------------


def test_build_from_paths_writes_parquet(tmp_path: Path) -> None:
    stations = _stations(
        [{"station_id": "s1", "first_seen": "2024-01-01", "last_seen": "2024-01-03"}]
    )
    fuel_daily = _fuel_daily(
        [
            {
                "station_id": "s1",
                "fuel_code": "U91",
                "date": "2024-01-01",
                "price_mean": 200.0,
                "price_min": 199.0,
                "price_max": 201.0,
                "n_obs": 3,
            }
        ]
    )
    s_path = tmp_path / "stations.parquet"
    f_path = tmp_path / "fuel_daily.parquet"
    out = tmp_path / "panel.parquet"
    stations.to_parquet(s_path, engine="pyarrow", compression="zstd", index=False)
    fuel_daily.to_parquet(f_path, engine="pyarrow", compression="zstd", index=False)

    pg.build_from_paths(s_path, f_path, out, span_start="2024-01-01")
    panel = pd.read_parquet(out)
    assert list(panel.columns) == list(pg.OUTPUT_COLUMNS)
    assert len(panel) == 6  # 3 days x 2 fuels


def test_missing_required_columns_raises() -> None:
    stations = _stations(
        [{"station_id": "s1", "first_seen": "2024-01-01", "last_seen": "2024-01-02"}]
    )
    bad_fuel = pd.DataFrame([{"date": "2024-01-01", "price_mean": 200.0, "n_obs": 1}])
    with pytest.raises(ValueError, match="missing columns"):
        pg.build(stations, bad_fuel, span_start="2024-01-01")
