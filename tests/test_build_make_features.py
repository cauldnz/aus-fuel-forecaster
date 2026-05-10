"""Hermetic tests for build.make_features.

Per CLAUDE.md "Test-first for feature engineering" — every block gets
unit tests with synthetic panels that pin lag / window / null behaviour.
Bugs in these blocks are silent and devastating.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fuel_pred.build import make_features as mf

# ----------------------------- Fixtures -----------------------------


def _panel(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


@pytest.fixture
def panel_two_stations_one_fuel() -> pd.DataFrame:
    """40 daily rows of U91 prices for 2 stations.

    Long enough to validate 28-day rolling windows.
    """
    dates = pd.date_range("2024-01-01", periods=40, freq="D").date
    rows = []
    for sid in ("s1", "s2"):
        for i, d in enumerate(dates):
            base = 200.0 if sid == "s1" else 215.0
            rows.append(
                {
                    "station_id": sid,
                    "fuel_code": "U91",
                    "date": d,
                    "price_mean": base + (i % 7),
                    "price_min": base + (i % 7) - 1,
                    "price_max": base + (i % 7) + 1,
                    "n_obs": 3,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def panel_with_diesel() -> pd.DataFrame:
    """Both U91 and DL rows for one station, for cross-fuel tests."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D").date
    rows = []
    for fuel, base in (("U91", 200.0), ("DL", 220.0)):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "station_id": "s1",
                    "fuel_code": fuel,
                    "date": d,
                    "price_mean": base + i,
                    "price_min": base + i - 0.5,
                    "price_max": base + i + 0.5,
                    "n_obs": 3,
                }
            )
    return pd.DataFrame(rows)


# ============================================================
# 7.1 Lag block
# ============================================================


def test_lag_features_have_expected_columns(panel_two_stations_one_fuel: pd.DataFrame) -> None:
    out = mf.add_lag_features(panel_two_stations_one_fuel)
    expected = {
        "lag_price_1",
        "lag_price_2",
        "lag_price_3",
        "lag_price_7",
        "lag_price_14",
        "lag_price_28",
        "roll_price_mean_7",
        "roll_price_mean_14",
        "roll_price_mean_28",
        "roll_price_std_7",
        "roll_price_std_14",
        "days_since_last_price_change",
        "price_minus_28d_min",
        "price_minus_28d_max",
        "xfuel_dl_price_lag_0",
        "xfuel_dl_price_lag_1",
        "xfuel_u91_minus_dl_lag_1",
        "xfuel_dl_roll_mean_7",
    }
    assert expected <= set(out.columns)


def test_lag_n_is_value_n_days_back(panel_two_stations_one_fuel: pd.DataFrame) -> None:
    """`lag_price_3` at row i must equal `price_mean` at row i-3 (within group)."""
    out = mf.add_lag_features(panel_two_stations_one_fuel)
    s1 = out[out["station_id"] == "s1"].sort_values("date").reset_index(drop=True)
    # First 3 rows have no value to look back to.
    assert s1["lag_price_3"].iloc[:3].isna().all()
    # From row 3 onward, lag_price_3 = price_mean shifted back by 3.
    for i in range(3, len(s1)):
        assert s1["lag_price_3"].iloc[i] == s1["price_mean"].iloc[i - 3]


def test_lag_does_not_cross_station_boundary(panel_two_stations_one_fuel: pd.DataFrame) -> None:
    """First row of station s2 must have null lag_price_1, not s1's last value."""
    out = mf.add_lag_features(panel_two_stations_one_fuel)
    s2 = out[out["station_id"] == "s2"].sort_values("date").reset_index(drop=True)
    assert pd.isna(s2["lag_price_1"].iloc[0])


def test_rolling_min_periods_avoids_early_life_leakage(
    panel_two_stations_one_fuel: pd.DataFrame,
) -> None:
    """`roll_price_mean_7` must be null for the first 7 rows (window=min_periods=7).

    Spec §7.1: "All rolling windows use min_periods=window".
    """
    out = mf.add_lag_features(panel_two_stations_one_fuel)
    s1 = out[out["station_id"] == "s1"].sort_values("date").reset_index(drop=True)
    # Index 0..6 (first 7 rows) — null because we need 7 prior obs for the
    # mean of price[t-7..t-1] (we shift before rolling).
    assert s1["roll_price_mean_7"].iloc[:7].isna().all()
    # Index 7 = first valid value.
    assert pd.notna(s1["roll_price_mean_7"].iloc[7])


def test_rolling_uses_past_only_no_future_leakage(
    panel_two_stations_one_fuel: pd.DataFrame,
) -> None:
    """`roll_price_mean_7` at row i must be the mean of price[i-7..i-1] —
    NOT including row i itself (that would be future-information leakage)."""
    out = mf.add_lag_features(panel_two_stations_one_fuel)
    s1 = out[out["station_id"] == "s1"].sort_values("date").reset_index(drop=True)
    # At index 7: mean of indices 0..6.
    expected = s1["price_mean"].iloc[0:7].mean()
    assert s1["roll_price_mean_7"].iloc[7] == pytest.approx(expected)


def test_price_minus_28d_min_max(panel_two_stations_one_fuel: pd.DataFrame) -> None:
    """Captures cycle phase: today's price minus past-28d min/max."""
    out = mf.add_lag_features(panel_two_stations_one_fuel)
    s1 = out[out["station_id"] == "s1"].sort_values("date").reset_index(drop=True)
    # Index 28: price[28] minus min(price[0..27]).
    expected_min = s1["price_mean"].iloc[28] - s1["price_mean"].iloc[0:28].min()
    assert s1["price_minus_28d_min"].iloc[28] == pytest.approx(expected_min)


def test_xfuel_dl_lag_0_equals_diesel_price(panel_with_diesel: pd.DataFrame) -> None:
    """U91 row at date X gets xfuel_dl_price_lag_0 = DL price at X."""
    out = mf.add_lag_features(panel_with_diesel)
    u91 = out[out["fuel_code"] == "U91"].sort_values("date").reset_index(drop=True)
    dl_only = panel_with_diesel[panel_with_diesel["fuel_code"] == "DL"]
    dl = dl_only.sort_values("date").reset_index(drop=True)
    for i in range(len(u91)):
        assert u91["xfuel_dl_price_lag_0"].iloc[i] == pytest.approx(dl["price_mean"].iloc[i])


def test_xfuel_columns_null_on_diesel_rows(panel_with_diesel: pd.DataFrame) -> None:
    """xfuel_* features are populated only on U91 rows."""
    out = mf.add_lag_features(panel_with_diesel)
    dl = out[out["fuel_code"] == "DL"]
    for col in ("xfuel_dl_price_lag_0", "xfuel_dl_price_lag_1",
                "xfuel_u91_minus_dl_lag_1", "xfuel_dl_roll_mean_7"):
        assert dl[col].isna().all()


def test_xfuel_u91_minus_dl_spread(panel_with_diesel: pd.DataFrame) -> None:
    """Spread = lag_price_1 (U91) - xfuel_dl_price_lag_1 (DL)."""
    out = mf.add_lag_features(panel_with_diesel)
    u91 = out[out["fuel_code"] == "U91"].sort_values("date").reset_index(drop=True)
    # Row 1: U91 lag_1 = U91 price at day 0 = 200. DL lag_1 = DL price at day 0 = 220.
    assert u91["xfuel_u91_minus_dl_lag_1"].iloc[1] == pytest.approx(200.0 - 220.0)


# ============================================================
# 7.2 Upstream block
# ============================================================


def test_upstream_features_lag_brent_and_audusd() -> None:
    panel = _panel(
        [
            {
                "station_id": "s1", "fuel_code": "U91", "date": "2024-01-08",
                "price_mean": 200.0, "price_min": 199.0, "price_max": 201.0, "n_obs": 1,
            },
        ]
    )
    brent = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=10, freq="D").date,
         "close": [80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0]}
    )
    audusd = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=10, freq="D").date,
         "audusd": [0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74]}
    )

    out = mf.add_upstream_features(panel, brent, audusd)
    # 2024-01-08 is index 7 in the daily series. lag_0 = day 7's value.
    assert out["upstream_brent_lag_0"].iloc[0] == pytest.approx(87.0)
    assert out["upstream_audusd_lag_0"].iloc[0] == pytest.approx(0.72)
    # lag_7 = day 0's value.
    assert out["upstream_brent_lag_7"].iloc[0] == pytest.approx(80.0)


def test_upstream_brent_aud_is_ratio() -> None:
    panel = _panel(
        [
            {
                "station_id": "s1", "fuel_code": "U91", "date": "2024-01-01",
                "price_mean": 200.0, "price_min": 199.0, "price_max": 201.0, "n_obs": 1,
            },
        ]
    )
    brent = pd.DataFrame({"date": [dt.date(2024, 1, 1)], "close": [80.0]})
    audusd = pd.DataFrame({"date": [dt.date(2024, 1, 1)], "audusd": [0.65]})

    out = mf.add_upstream_features(panel, brent, audusd)
    assert out["upstream_brent_aud_lag_0"].iloc[0] == pytest.approx(80.0 / 0.65)


def test_upstream_tgp_columns_null_when_aip_missing() -> None:
    """Without an aip_tgp arg, the TGP columns ship as null per spec §7.2."""
    panel = _panel(
        [{"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01", "price_mean": 200.0,
          "price_min": 199.0, "price_max": 201.0, "n_obs": 1}]
    )
    brent = pd.DataFrame({"date": [dt.date(2024, 1, 1)], "close": [80.0]})
    audusd = pd.DataFrame({"date": [dt.date(2024, 1, 1)], "audusd": [0.65]})
    out = mf.add_upstream_features(panel, brent, audusd)
    for col in ("upstream_tgp_sydney_lag_0", "upstream_tgp_sydney_lag_3",
                "upstream_tgp_sydney_lag_7", "upstream_tgp_minus_brent_aud_lag_7"):
        assert out[col].isna().all()


def test_upstream_tgp_populates_when_aip_provided() -> None:
    """When aip_tgp is passed, upstream_tgp_sydney_lag_* should be set."""
    panel = _panel(
        [
            {
                "station_id": "s1", "fuel_code": "U91", "date": "2024-01-15",
                "price_mean": 200.0, "price_min": 199.0, "price_max": 201.0, "n_obs": 1,
            }
        ]
    )
    brent = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=15, freq="D").date,
         "close": list(range(80, 95))}
    )
    audusd = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=15, freq="D").date,
         "audusd": [0.65 + 0.001 * i for i in range(15)]}
    )
    aip = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=15, freq="D").date,
         "ulp_sydney": [170.0 + i for i in range(15)],
         "diesel_sydney": [190.0 + i for i in range(15)]}
    )

    out = mf.add_upstream_features(panel, brent, audusd, aip_tgp=aip)
    # 2024-01-15 lag_0 = 184; lag_7 = 177.
    assert float(out["upstream_tgp_sydney_lag_0"].iloc[0]) == pytest.approx(184.0)
    assert float(out["upstream_tgp_sydney_lag_7"].iloc[0]) == pytest.approx(177.0)
    # Margin proxy: tgp_lag_7 - brent_aud_lag_7. Both non-null.
    assert pd.notna(out["upstream_tgp_minus_brent_aud_lag_7"].iloc[0])


# ============================================================
# 7.3 Calendar block
# ============================================================


def test_day_of_week_correct() -> None:
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01"},  # Monday
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-07"},  # Sunday
    ])
    out = mf.add_calendar_features(panel)
    assert int(out["cal_day_of_week"].iloc[0]) == 0
    assert int(out["cal_day_of_week"].iloc[1]) == 6


def test_day_of_fortnight_anchor() -> None:
    """Anchor 2016-07-04 (Monday). Itself = 0; 14 days later = 0 again."""
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2016-07-04"},  # anchor
        {"station_id": "s1", "fuel_code": "U91", "date": "2016-07-05"},
        {"station_id": "s1", "fuel_code": "U91", "date": "2016-07-18"},  # +14 days
    ])
    out = mf.add_calendar_features(panel)
    assert int(out["cal_day_of_fortnight"].iloc[0]) == 0
    assert int(out["cal_day_of_fortnight"].iloc[1]) == 1
    assert int(out["cal_day_of_fortnight"].iloc[2]) == 0


def test_public_holiday_flag_and_distances() -> None:
    """2024-01-01 is New Year's Day (NSW public holiday)."""
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2023-12-30"},
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01"},
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-02"},
    ])
    out = mf.add_calendar_features(panel)
    assert bool(out["cal_is_public_holiday"].iloc[1]) is True
    # NYD itself: distance = 0 in both directions.
    assert int(out["cal_days_to_next_public_holiday"].iloc[1]) == 0
    assert int(out["cal_days_since_last_public_holiday"].iloc[1]) == 0
    # 2024-01-02: 0 days since (NYD), positive days to next.
    assert int(out["cal_days_since_last_public_holiday"].iloc[2]) == 1
    assert int(out["cal_days_to_next_public_holiday"].iloc[2]) > 0


def test_school_holiday_flag(tmp_path: Path) -> None:
    """A date inside a term is school_holiday=False; outside = True."""
    school_terms_path = tmp_path / "terms.csv"
    school_terms_path.write_text(
        "year,term,start_date,end_date,division\n"
        "2024,1,2024-01-30,2024-04-12,eastern\n"
    )
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-15"},  # before term 1
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-02-15"},  # in term 1
    ])
    out = mf.add_calendar_features(panel, school_terms_path=school_terms_path)
    assert bool(out["cal_is_school_holiday_nsw"].iloc[0]) is True
    assert bool(out["cal_is_school_holiday_nsw"].iloc[1]) is False


def test_first_business_day_after_break() -> None:
    """Monday after a normal weekend is True; Tuesday is False."""
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-02-05"},  # Monday after weekend
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-02-06"},  # Tuesday
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-02-03"},  # Saturday
    ])
    out = mf.add_calendar_features(panel)
    assert bool(out["cal_is_first_business_day_after_break"].iloc[0]) is True
    assert bool(out["cal_is_first_business_day_after_break"].iloc[1]) is False
    assert bool(out["cal_is_first_business_day_after_break"].iloc[2]) is False  # weekend itself


# ============================================================
# 7.4 Context block
# ============================================================


def test_context_distances_and_radius() -> None:
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01"},
    ])
    top_n = pd.DataFrame([
        {"station_id": "s1", "counter_rank": 1, "counter_id": "c1", "distance_km": 1.0},
        {"station_id": "s1", "counter_rank": 2, "counter_id": "c2", "distance_km": 2.0},
        {"station_id": "s1", "counter_rank": 3, "counter_id": "c3", "distance_km": 3.0},
    ])
    summary = pd.DataFrame([
        {"station_id": "s1", "stn_n_counters_within_5km": 5}
    ])
    traffic_daily = pd.DataFrame()  # empty: lags become null

    out = mf.add_context_features(panel, top_n, summary, traffic_daily)
    assert float(out["ctx_traffic_top1_distance_km"].iloc[0]) == 1.0
    assert float(out["ctx_traffic_top2_distance_km"].iloc[0]) == 2.0
    assert float(out["ctx_traffic_top3_distance_km"].iloc[0]) == 3.0
    assert int(out["ctx_traffic_5km_radius_count"].iloc[0]) == 5
    # Phase 5 macro columns: null when their upstream isn't passed.
    for col in (
        "ctx_inflation_expectations_lag_7",
        "ctx_asx200_lag_1",
        "ctx_cash_rate",
    ):
        assert out[col].isna().all()


def test_context_macro_features_populate_from_phase5_inputs() -> None:
    """When cash_rate / asx200 / inflation_expectations are passed,
    their values flow into the corresponding ctx_ columns with the
    right lag and forward-fill semantics."""
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-15"},
    ])
    top_n = pd.DataFrame([
        {"station_id": "s1", "counter_rank": 1, "counter_id": "c1", "distance_km": 1.0},
    ])
    summary = pd.DataFrame([{"station_id": "s1", "stn_n_counters_within_5km": 1}])
    cash_rate = pd.DataFrame(
        {"date": [dt.date(2024, 1, 1)], "cash_rate": [4.35]},
    )
    asx200 = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-10", "2024-01-15", freq="D").date,
            "close": [7400.0, 7410.0, 7420.0, 7430.0, 7440.0, 7450.0],
        }
    )
    inflation_expectations = pd.DataFrame(
        {"date": [dt.date(2023, 12, 31)], "inflation_expectations": [4.5]},
    )

    out = mf.add_context_features(
        panel, top_n, summary, pd.DataFrame(),
        cash_rate=cash_rate,
        asx200=asx200,
        inflation_expectations=inflation_expectations,
    )
    # Cash rate forward-fills from 2024-01-01 → 2024-01-15.
    assert float(out["ctx_cash_rate"].iloc[0]) == pytest.approx(4.35)
    # ASX 200 lag 1: 2024-01-15 → close at 2024-01-14 = 7440.
    assert float(out["ctx_asx200_lag_1"].iloc[0]) == pytest.approx(7440.0)
    # Inflation expectations lag 7: 2024-01-15 minus 7 days = 2024-01-08;
    # forward-filled from 2023-12-31 → 4.5.
    assert float(out["ctx_inflation_expectations_lag_7"].iloc[0]) == pytest.approx(4.5)


def test_context_50km_cutoff_nulls_traffic_columns() -> None:
    """If the closest counter is > 50 km, all ctx_traffic_* columns are null."""
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01"},
    ])
    top_n = pd.DataFrame([
        {"station_id": "s1", "counter_rank": 1, "counter_id": "c1", "distance_km": 60.0},
        {"station_id": "s1", "counter_rank": 2, "counter_id": "c2", "distance_km": 70.0},
        {"station_id": "s1", "counter_rank": 3, "counter_id": "c3", "distance_km": 80.0},
    ])
    summary = pd.DataFrame([
        {"station_id": "s1", "stn_n_counters_within_5km": 0}
    ])
    out = mf.add_context_features(panel, top_n, summary, pd.DataFrame())
    for col in out.columns:
        if col.startswith("ctx_traffic_"):
            assert out[col].isna().all(), f"col {col} should be null but isn't"


def test_context_traffic_lag_joins_to_counter_daily_total() -> None:
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-08"},
    ])
    top_n = pd.DataFrame([
        {"station_id": "s1", "counter_rank": 1, "counter_id": "c1", "distance_km": 1.0},
    ])
    summary = pd.DataFrame([{"station_id": "s1", "stn_n_counters_within_5km": 1}])
    traffic_daily = pd.DataFrame(
        {
            "counter_id": ["c1"] * 8,
            "date": pd.date_range("2024-01-01", periods=8, freq="D").date,
            "daily_total": [100, 110, 120, 130, 140, 150, 160, 170],
        }
    )
    out = mf.add_context_features(panel, top_n, summary, traffic_daily)
    # 2024-01-08, lag_1 = 2024-01-07 = 160. lag_7 = 2024-01-01 = 100.
    assert int(out["ctx_traffic_top1_lag_1"].iloc[0]) == 160
    assert int(out["ctx_traffic_top1_lag_7"].iloc[0]) == 100


# ============================================================
# 7.5 Station block
# ============================================================


def test_station_features_brand_columns() -> None:
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01"},
    ])
    stations = pd.DataFrame(
        [
            {
                "station_id": "s1",
                "brand_raw": "EG Ampol",
                "brand_canonical": "Ampol",
                "brand_is_major": True,
                "lat": -33.93,
                "lon": 151.20,
                "sa2_name": "Sydney - Eastern Suburbs",
            }
        ]
    )
    out = mf.add_station_features(panel, stations)
    assert out["stn_brand_raw"].iloc[0] == "EG Ampol"
    assert out["stn_brand_canonical"].iloc[0] == "Ampol"
    assert bool(out["stn_brand_is_major"].iloc[0]) is True


def test_station_metro_heuristic() -> None:
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01"},
        {"station_id": "s2", "fuel_code": "U91", "date": "2024-01-01"},
    ])
    stations = pd.DataFrame([
        {"station_id": "s1", "lat": -33.93, "lon": 151.20, "sa2_name": "Sydney - Inner West"},
        {"station_id": "s2", "lat": -34.55, "lon": 150.37, "sa2_name": "Bombala"},
    ])
    out = mf.add_station_features(panel, stations)
    by_id = {row["station_id"]: row for _, row in out.iterrows()}
    assert bool(by_id["s1"]["stn_is_metro"]) is True
    assert bool(by_id["s2"]["stn_is_metro"]) is False


def test_station_competitor_counts() -> None:
    """Three stations: s1+s2 within 2 km, s3 far away."""
    panel = _panel([
        {"station_id": sid, "fuel_code": "U91", "date": "2024-01-01"}
        for sid in ("s1", "s2", "s3")
    ])
    stations = pd.DataFrame([
        {"station_id": "s1", "lat": -33.930, "lon": 151.200, "sa2_name": "Mascot"},
        {"station_id": "s2", "lat": -33.931, "lon": 151.201, "sa2_name": "Mascot"},
        {"station_id": "s3", "lat": -34.420, "lon": 150.890, "sa2_name": "Wollongong"},
    ])
    out = mf.add_station_features(panel, stations)
    by_id = {row["station_id"]: row for _, row in out.iterrows()}
    assert int(by_id["s1"]["stn_competitors_within_2km"]) == 1  # s2
    assert int(by_id["s3"]["stn_competitors_within_2km"]) == 0


def test_station_is_franchisee_stub_null() -> None:
    panel = _panel([{"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01"}])
    stations = pd.DataFrame([
        {"station_id": "s1", "brand_raw": "EG Ampol", "lat": -33.93, "lon": 151.20, "sa2_name": "X"}
    ])
    out = mf.add_station_features(panel, stations)
    # Per spec §13 Q3, stays null pending the cross-brand research pass.
    assert out["stn_is_franchisee"].isna().all()


# ============================================================
# 7.6 Weather block
# ============================================================


def test_weather_features_join_per_station(tmp_path: Path) -> None:
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01"},
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-02"},
        {"station_id": "s2", "fuel_code": "U91", "date": "2024-01-01"},
    ])
    weather_dir = tmp_path / "weather"
    weather_dir.mkdir()
    pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="D").date,
        "wx_temp_max_c": [25.0, 26.0, 27.0],
        "wx_temp_min_c": [15.0, 16.0, 17.0],
        "wx_precipitation_mm": [0.0, 1.0, 2.0],
        "wx_wind_speed_max_kmh": [10.0, 11.0, 12.0],
        "wx_weather_code": [0, 1, 2],
    }).to_parquet(weather_dir / "s1.parquet", engine="pyarrow", compression="zstd", index=False)

    out = mf.add_weather_features(panel, weather_dir)
    s1_jan1 = out[(out["station_id"] == "s1") & (out["date"] == dt.date(2024, 1, 1))].iloc[0]
    assert s1_jan1["wx_temp_max_c"] == 25.0
    # s2 has no parquet — wx columns are null.
    s2_jan1 = out[out["station_id"] == "s2"].iloc[0]
    assert pd.isna(s2_jan1["wx_temp_max_c"])


def test_weather_missing_dir_yields_null_columns(tmp_path: Path) -> None:
    panel = _panel([{"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01"}])
    out = mf.add_weather_features(panel, tmp_path / "nope")
    for col in ("wx_temp_max_c", "wx_temp_min_c", "wx_precipitation_mm",
                "wx_wind_speed_max_kmh", "wx_weather_code"):
        assert col in out.columns
        assert out[col].isna().all()


# ============================================================
# 7.7 Demographic block
# ============================================================


def test_sa2_features_join_from_stations() -> None:
    panel = _panel([
        {"station_id": "s1", "fuel_code": "U91", "date": "2024-01-01"},
    ])
    stations = pd.DataFrame([{
        "station_id": "s1",
        "sa2_total_population": 21573,
        "sa2_median_age": 30,
        "sa2_median_household_income_weekly": 1900,
        "sa2_seifa_irsd_score": 1098,
    }])
    out = mf.add_sa2_features(panel, stations)
    assert int(out["sa2_total_population"].iloc[0]) == 21573
    # Deferred derived columns exist as nulls.
    for col in ("sa2_pct_drive_to_work", "sa2_motor_vehicles_per_dwelling",
                "sa2_pct_renters", "sa2_pct_employed_full_time",
                "sa2_pct_aged_65_plus", "sa2_pct_one_parent_family"):
        assert col in out.columns
        assert out[col].isna().all()


# ============================================================
# 7.8 Targets
# ============================================================


def test_targets_y_t1_is_next_day_price(panel_two_stations_one_fuel: pd.DataFrame) -> None:
    out = mf.add_targets(panel_two_stations_one_fuel)
    s1 = out[out["station_id"] == "s1"].sort_values("date").reset_index(drop=True)
    # y_t1 at row i = price at row i+1.
    for i in range(len(s1) - 1):
        assert s1["y_t1"].iloc[i] == s1["price_mean"].iloc[i + 1]
    # End of series: last row's y_t1 is null.
    assert pd.isna(s1["y_t1"].iloc[-1])


def test_targets_only_on_u91_rows(panel_with_diesel: pd.DataFrame) -> None:
    out = mf.add_targets(panel_with_diesel)
    assert out[out["fuel_code"] == "DL"]["y_t1"].isna().all()
    # U91 has at least some non-null targets (everything except the last row).
    u91_targets = out[out["fuel_code"] == "U91"]["y_t1"]
    assert u91_targets.notna().sum() == len(panel_with_diesel) // 2 - 1


def test_targets_y_t1_t7_mean(panel_two_stations_one_fuel: pd.DataFrame) -> None:
    out = mf.add_targets(panel_two_stations_one_fuel)
    s1 = out[out["station_id"] == "s1"].sort_values("date").reset_index(drop=True)
    # At row i, y_t1_t7 = mean(price[i+1..i+7]).
    expected = s1["price_mean"].iloc[1:8].mean()
    assert s1["y_t1_t7"].iloc[0] == pytest.approx(expected)


# ============================================================
# Orchestrator
# ============================================================


def test_make_features_orchestrator_produces_expected_blocks(
    panel_with_diesel: pd.DataFrame, tmp_path: Path
) -> None:
    """End-to-end smoke through the orchestrator with minimal fixtures."""
    brent = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D").date,
        "close": np.linspace(80.0, 89.0, 10),
    })
    audusd = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D").date,
        "audusd": np.linspace(0.65, 0.74, 10),
    })
    stations = pd.DataFrame([{
        "station_id": "s1",
        "brand_raw": "Ampol",
        "brand_canonical": "Ampol",
        "brand_is_major": True,
        "lat": -33.93,
        "lon": 151.20,
        "sa2_name": "Sydney - Inner",
        "sa2_total_population": 21573,
        "sa2_median_age": 30,
        "sa2_median_household_income_weekly": 1900,
        "sa2_seifa_irsd_score": 1098,
    }])
    top_n = pd.DataFrame([
        {"station_id": "s1", "counter_rank": 1, "counter_id": "c1", "distance_km": 1.0},
        {"station_id": "s1", "counter_rank": 2, "counter_id": "c2", "distance_km": 2.0},
        {"station_id": "s1", "counter_rank": 3, "counter_id": "c3", "distance_km": 3.0},
    ])
    summary = pd.DataFrame([{"station_id": "s1", "stn_n_counters_within_5km": 3}])
    traffic_daily = pd.DataFrame()

    features = mf.make_features(
        panel_with_diesel,
        brent=brent,
        audusd=audusd,
        stations=stations,
        top_n=top_n,
        summary=summary,
        traffic_daily=traffic_daily,
        weather_dir=None,
    )

    # All expected block prefixes are present.
    expected_prefixes = ("lag_", "roll_", "xfuel_", "upstream_", "cal_",
                        "ctx_", "stn_", "wx_", "sa2_", "y_")
    for prefix in expected_prefixes:
        matching = [c for c in features.columns if c.startswith(prefix)]
        assert matching, f"no columns with prefix {prefix!r}"

    # Targets respect the U91-only rule.
    dl_rows = features[features["fuel_code"] == "DL"]
    assert dl_rows["y_t1"].isna().all()


def test_make_features_from_paths_writes_parquet(
    panel_with_diesel: pd.DataFrame, tmp_path: Path
) -> None:
    """Round-trip through the file-IO wrapper."""
    panel_path = tmp_path / "panel.parquet"
    panel_with_diesel.to_parquet(panel_path, engine="pyarrow", compression="zstd", index=False)

    pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D").date,
        "close": np.linspace(80.0, 89.0, 10),
    }).to_parquet(tmp_path / "brent.parquet", engine="pyarrow", compression="zstd", index=False)
    pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D").date,
        "audusd": np.linspace(0.65, 0.74, 10),
    }).to_parquet(tmp_path / "audusd.parquet", engine="pyarrow", compression="zstd", index=False)
    pd.DataFrame([{
        "station_id": "s1", "brand_raw": "Ampol", "brand_canonical": "Ampol",
        "brand_is_major": True, "lat": -33.93, "lon": 151.20, "sa2_name": "Mascot",
    }]).to_parquet(tmp_path / "stations.parquet", engine="pyarrow", compression="zstd", index=False)
    pd.DataFrame([
        {"station_id": "s1", "counter_rank": 1, "counter_id": "c1", "distance_km": 1.0},
    ]).to_parquet(tmp_path / "top_n.parquet", engine="pyarrow", compression="zstd", index=False)
    pd.DataFrame([{"station_id": "s1", "stn_n_counters_within_5km": 1}]).to_parquet(
        tmp_path / "summary.parquet", engine="pyarrow", compression="zstd", index=False
    )

    out = tmp_path / "features.parquet"
    mf.make_features_from_paths(
        panel_path, out,
        brent_path=tmp_path / "brent.parquet",
        audusd_path=tmp_path / "audusd.parquet",
        stations_path=tmp_path / "stations.parquet",
        top_n_path=tmp_path / "top_n.parquet",
        summary_path=tmp_path / "summary.parquet",
        traffic_daily_path=tmp_path / "missing.parquet",
        weather_dir=None,
        school_terms_path=None,
    )

    df = pd.read_parquet(out)
    assert len(df) == len(panel_with_diesel)
    assert "y_t1" in df.columns
