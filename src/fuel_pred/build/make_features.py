"""Build the training-ready feature matrix per spec §7.

Reads:
- ``data/interim/panel.parquet`` (from `build.panel_grid`)
- ``data/raw/brent.parquet``, ``data/raw/audusd.parquet``
- ``data/interim/stations.parquet`` (post Phase 3 — has lat/lon, sa2_*, brand_*)
- ``data/interim/station_to_counter.parquet`` + summary (from `spatial.nearest`)
- ``data/interim/traffic_daily.parquet`` (from `clean.traffic`)
- ``data/raw/weather/<station_id>.parquet`` (from `fetch.weather`; optional)
- ``data/static/nsw_school_terms.csv``

Writes:
- ``data/processed/features.parquet``

Each feature block is a pure function ``add_<block>_features(df, ...) → df``
per spec §7, so individual blocks can be ablated for experimentation.

Spec: spec.md §7.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from fuel_pred import config

logger = logging.getLogger(__name__)

# Day-of-fortnight anchor — spec §7.3, set in config.
DOF_ANCHOR_DATE: dt.date = dt.date.fromisoformat(config.DOF_ANCHOR)

# Static metro-suburb prefixes used by `add_station_features` heuristic
# until the augmentor exposes a UCL/SOS field — see spec §7.5 amendment.
METRO_SA2_PREFIXES: tuple[str, ...] = (
    "Sydney - ",
    "Newcastle",
    "Wollongong",
    "Central Coast",
    "Lake Macquarie",
)

# Phase-5 columns: populated when their fetcher's parquet is present,
# null otherwise. Each upstream is independently optional — the
# `_add_macro_feature` helper handles None/missing inputs gracefully.
# `ctx_consumer_confidence_lag_7` is replaced by
# `ctx_inflation_expectations_lag_7` (RBA G3) per spec §5.2 / §7.4
# — Roy Morgan doesn't publish a clean machine-readable feed.


# ============================================================
# 7.1 Lag block
# ============================================================


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per `(station_id, fuel_code)`: lags, rolling stats, gap counters.

    Includes cross-fuel features for U91 rows (Diesel price joined on
    `(station_id, date)`). All rolling windows use ``min_periods=window``
    to avoid early-life leakage per spec §7.1.
    """
    out = df.sort_values(["station_id", "fuel_code", "date"]).reset_index(drop=True)
    grouped = out.groupby(["station_id", "fuel_code"], sort=False, observed=True)

    # Plain lag values.
    for n in (1, 2, 3, 7, 14, 28):
        out[f"lag_price_{n}"] = grouped["price_mean"].shift(n)

    # Rolling means + stds. min_periods=window prevents leakage.
    for window in (7, 14, 28):
        out[f"roll_price_mean_{window}"] = grouped["price_mean"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean()
        )
    for window in (7, 14):
        out[f"roll_price_std_{window}"] = grouped["price_mean"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=w).std()
        )

    # 28-day relative-position features: today's price vs the past-28d window.
    for op_name, op in (("min", "min"), ("max", "max")):
        roll = grouped["price_mean"].transform(
            lambda s, fn=op: s.shift(1).rolling(28, min_periods=28).agg(fn)
        )
        out[f"price_minus_28d_{op_name}"] = out["price_mean"] - roll

    # Days since last price change. Within (station, fuel), reset counter
    # whenever the price moves; carry the gap forward.
    out["days_since_last_price_change"] = (
        grouped["price_mean"]
        .transform(_days_since_last_change)
        .astype("Float64")
    )

    # Cross-fuel features (Diesel price joined onto U91 rows).
    out = _add_cross_fuel_features(out)

    return out


def _days_since_last_change(prices: pd.Series) -> pd.Series:
    """Count rows since the last time `price` changed (inclusive at change=0)."""
    # NaN-safe diff; True where the price differs from the previous row.
    diffs = prices.ffill().diff().fillna(0) != 0
    # Number of rows since the last True (counting from 0).
    counter = []
    days = float("nan")  # before any obs, undefined
    for is_change in diffs:
        if pd.isna(is_change):
            counter.append(np.nan)
            continue
        if bool(is_change):
            days = 0.0
        elif not np.isnan(days):
            days += 1.0
        counter.append(days)
    return pd.Series(counter, index=prices.index)


def _add_cross_fuel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join Diesel `price_mean` columns onto U91 rows as `xfuel_dl_*`.

    Cross-fuel features are populated only on U91 rows; DL rows get
    these columns as null (per spec §7.1 — they exist solely for U91
    target rows).
    """
    diesel = df[df["fuel_code"] == "DL"][
        ["station_id", "date", "price_mean", "roll_price_mean_7", "lag_price_1"]
    ].rename(
        columns={
            "price_mean": "xfuel_dl_price_lag_0",
            "lag_price_1": "xfuel_dl_price_lag_1",
            "roll_price_mean_7": "xfuel_dl_roll_mean_7",
        }
    )

    # Merge onto every row, then null out the DL rows' own xfuel_* values
    # (so the column is populated on U91 rows only — clearer for the model).
    out = df.merge(diesel, on=["station_id", "date"], how="left")

    out["xfuel_u91_minus_dl_lag_1"] = out["lag_price_1"] - out["xfuel_dl_price_lag_1"]

    dl_mask = out["fuel_code"] == "DL"
    for col in (
        "xfuel_dl_price_lag_0",
        "xfuel_dl_price_lag_1",
        "xfuel_dl_roll_mean_7",
        "xfuel_u91_minus_dl_lag_1",
    ):
        out.loc[dl_mask, col] = pd.NA

    return out


# ============================================================
# 7.2 Upstream block
# ============================================================


def add_upstream_features(
    df: pd.DataFrame,
    brent: pd.DataFrame,
    audusd: pd.DataFrame,
    *,
    aip_tgp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Brent + AUD/USD + AIP TGP lags, ratios, and changes.

    Args:
        df: panel rows with at least a `date` column.
        brent: from fetch.brent (daily OHLC).
        audusd: from fetch.audusd (daily AUD/USD).
        aip_tgp: from fetch.aip_tgp (daily Sydney ULP + Diesel TGP).
            Optional — when None, the `upstream_tgp_*` columns are
            null per spec §7.2.
    """
    # Brent: take `close` as the daily series; forward-fill to cover
    # weekends/holidays so date-aligned joins always hit.
    brent_d = (
        brent.assign(date=pd.to_datetime(brent["date"]).dt.date)
        .sort_values("date")
        .set_index("date")[["close"]]
        .rename(columns={"close": "brent"})
    )
    audusd_d = (
        audusd.assign(date=pd.to_datetime(audusd["date"]).dt.date)
        .sort_values("date")
        .set_index("date")[["audusd"]]
    )

    # Build a continuous daily series from the upstream sources' own start
    # to the panel's end so lag_14 of the panel's earliest date is defined.
    upstream_start = min(brent_d.index.min(), audusd_d.index.min())
    panel_end = max(df["date"].max(), brent_d.index.max(), audusd_d.index.max())
    if aip_tgp is not None and not aip_tgp.empty:
        tgp_d = (
            aip_tgp.assign(date=pd.to_datetime(aip_tgp["date"]).dt.date)
            .sort_values("date")
            .set_index("date")[["ulp_sydney"]]
            .rename(columns={"ulp_sydney": "tgp_sydney"})
        )
        upstream_start = min(upstream_start, tgp_d.index.min())
        panel_end = max(panel_end, tgp_d.index.max())
    else:
        tgp_d = None

    full_dates = pd.date_range(upstream_start, panel_end, freq="D").date
    daily = pd.DataFrame(index=pd.Index(full_dates, name="date"))
    daily = daily.join(brent_d, how="left").join(audusd_d, how="left")
    if tgp_d is not None:
        daily = daily.join(tgp_d, how="left")
    daily = daily.ffill()

    # Lags off the daily-frequency series (not within station — these are global).
    for n in (0, 1, 3, 7, 14):
        daily[f"upstream_brent_lag_{n}"] = daily["brent"].shift(n)
    for n in (0, 1, 3, 7):
        daily[f"upstream_audusd_lag_{n}"] = daily["audusd"].shift(n)
    for n in (0, 7, 14):
        daily[f"upstream_brent_aud_lag_{n}"] = (
            daily[f"upstream_brent_lag_{n}"] / daily[f"upstream_audusd_lag_{min(n, 7)}"]
        )
    daily["upstream_brent_change_7d"] = daily["brent"] - daily["brent"].shift(7)
    daily["upstream_brent_change_14d"] = daily["brent"] - daily["brent"].shift(14)
    daily["upstream_audusd_change_7d"] = daily["audusd"] - daily["audusd"].shift(7)

    if tgp_d is not None:
        for n in (0, 3, 7):
            daily[f"upstream_tgp_sydney_lag_{n}"] = daily["tgp_sydney"].shift(n)
        # Margin proxy per spec §7.2: Sydney TGP minus
        # Brent / AUDUSD (i.e. retail-imported-cost spread, lag-7).
        daily["upstream_tgp_minus_brent_aud_lag_7"] = (
            daily["upstream_tgp_sydney_lag_7"] - daily["upstream_brent_aud_lag_7"]
        )
    else:
        for col in (
            "upstream_tgp_sydney_lag_0",
            "upstream_tgp_sydney_lag_3",
            "upstream_tgp_sydney_lag_7",
            "upstream_tgp_minus_brent_aud_lag_7",
        ):
            daily[col] = pd.NA

    upstream_cols = [c for c in daily.columns if c.startswith("upstream_")]
    out = df.merge(
        daily.reset_index()[["date", *upstream_cols]],
        on="date",
        how="left",
    )
    return out


# ============================================================
# 7.3 Calendar block
# ============================================================


def add_calendar_features(
    df: pd.DataFrame, school_terms_path: Path | None = None
) -> pd.DataFrame:
    """Day-of-week, month, day-of-fortnight, holidays, school-term flags."""
    out = df.copy()
    dt_col = pd.to_datetime(out["date"])

    out["cal_day_of_week"] = dt_col.dt.dayofweek.astype("Int64")
    out["cal_day_of_month"] = dt_col.dt.day.astype("Int64")
    out["cal_month"] = dt_col.dt.month.astype("Int64")
    out["cal_week_of_year"] = dt_col.dt.isocalendar().week.astype("Int64")
    out["cal_year"] = dt_col.dt.year.astype("Int64")

    # Day-of-fortnight, anchored at 2016-07-04 (spec §7.3).
    days_since_anchor = (dt_col.dt.date - DOF_ANCHOR_DATE).apply(lambda td: td.days)
    out["cal_day_of_fortnight"] = (days_since_anchor % 14).astype("Int64")

    # Public holidays: NSW.
    span_years = list(range(int(out["cal_year"].min()), int(out["cal_year"].max()) + 1))
    nsw_holidays = holidays.country_holidays("AU", subdiv="NSW", years=span_years)
    holiday_dates = sorted(nsw_holidays.keys())

    out["cal_is_public_holiday"] = dt_col.dt.date.isin(set(holiday_dates))
    out["cal_days_to_next_public_holiday"] = _days_until_next(dt_col.dt.date, holiday_dates)
    out["cal_days_since_last_public_holiday"] = _days_since_last(dt_col.dt.date, holiday_dates)

    # NSW school holidays from the static file.
    if school_terms_path is None:
        school_terms_path = config.DATA_STATIC / "nsw_school_terms.csv"
    school_term_dates = _load_school_term_dates(school_terms_path)
    in_term = dt_col.dt.date.apply(lambda d: _date_in_any_range(d, school_term_dates))
    out["cal_is_school_holiday_nsw"] = ~in_term

    # First-business-day-after-break flag.
    out["cal_is_first_business_day_after_break"] = _first_business_day_after_break(
        dt_col.dt.date.tolist(), set(holiday_dates)
    )

    return out


def _days_until_next(dates: pd.Series, holiday_dates: list[dt.date]) -> pd.Series:
    """Days until the next public holiday (inclusive 0 on holiday day itself)."""
    holiday_arr = np.array([d.toordinal() for d in holiday_dates])
    out = np.empty(len(dates), dtype=np.float64)
    for i, d in enumerate(dates):
        ordinal = d.toordinal()
        future = holiday_arr[holiday_arr >= ordinal]
        out[i] = float(future[0] - ordinal) if future.size else np.nan
    return pd.Series(out).astype("Float64")


def _days_since_last(dates: pd.Series, holiday_dates: list[dt.date]) -> pd.Series:
    """Days since the last public holiday (0 on holiday day itself)."""
    holiday_arr = np.array([d.toordinal() for d in holiday_dates])
    out = np.empty(len(dates), dtype=np.float64)
    for i, d in enumerate(dates):
        ordinal = d.toordinal()
        past = holiday_arr[holiday_arr <= ordinal]
        out[i] = float(ordinal - past[-1]) if past.size else np.nan
    return pd.Series(out).astype("Float64")


def _load_school_term_dates(path: Path) -> list[tuple[dt.date, dt.date]]:
    """Load (start, end) pairs from the static school-terms file."""
    if not path.exists():
        logger.warning("school terms file %s not found — skipping school flag", path)
        return []
    pairs: list[tuple[dt.date, dt.date]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = ["year", "term", "start_date", "end_date", "division"]
            row = next(csv.DictReader([line], fieldnames=fields))
            try:
                start = dt.date.fromisoformat(row["start_date"])
                end = dt.date.fromisoformat(row["end_date"])
            except (ValueError, TypeError):
                continue
            # Filter to eastern division per spec §12 Phase 2.
            if row.get("division", "eastern").strip() == "eastern":
                pairs.append((start, end))
    return pairs


def _date_in_any_range(date: dt.date, ranges: list[tuple[dt.date, dt.date]]) -> bool:
    return any(start <= date <= end for start, end in ranges)


def _first_business_day_after_break(dates: list[dt.date], holidays_set: set[dt.date]) -> list[bool]:
    """True for dates that are the first weekday after a weekend or holiday."""
    out: list[bool] = []
    for d in dates:
        if d.weekday() >= 5 or d in holidays_set:
            out.append(False)
            continue
        # Walk back day by day; if any of the previous days is a weekend
        # or holiday and the day before that ALSO wasn't a business day,
        # we're the first business day after a break.
        prev = d - dt.timedelta(days=1)
        out.append(prev.weekday() >= 5 or prev in holidays_set)
    return out


# ============================================================
# 7.4 Context block
# ============================================================


def _normalise_traffic_daily(traffic_daily: pd.DataFrame) -> pd.DataFrame | None:
    """Coerce traffic_daily to a sorted (counter_id, date, daily_total) frame.

    Returns None if the input is empty or missing required columns —
    callers should fall back to null traffic features in that case.
    """
    if traffic_daily.empty or "daily_total" not in traffic_daily.columns:
        return None
    df = traffic_daily.copy()
    if "station_key" in df.columns and "counter_id" not in df.columns:
        df = df.rename(columns={"station_key": "counter_id"})
    if "counter_id" not in df.columns:
        logger.warning("traffic_daily has no counter_id/station_key; skipping lag join")
        return None
    df["counter_id"] = df["counter_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["counter_id", "date", "daily_total"]].sort_values(["counter_id", "date"])


def add_context_features(
    df: pd.DataFrame,
    top_n_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    traffic_daily: pd.DataFrame,
    *,
    cash_rate: pd.DataFrame | None = None,
    asx200: pd.DataFrame | None = None,
    inflation_expectations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Top-N traffic counters + radius count + Phase 5 macro joins.

    Args:
        df: panel rows.
        top_n_table: from spatial.nearest top-N.
        summary_table: from spatial.nearest summary (radius count).
        traffic_daily: from clean.traffic.
        cash_rate: from fetch.cash_rate (monthly RBA F1.1). Optional —
            forward-filled to daily. None → `ctx_cash_rate` is null.
        asx200: from fetch.asx200 (daily yfinance ^AXJO). Optional —
            None → `ctx_asx200_lag_1` is null.
        inflation_expectations: from fetch.inflation_expectations
            (quarterly RBA G3 GCONEXP). Optional — forward-filled to
            daily; lagged 7. None → column is null. Substitution for
            ANZ-Roy Morgan Consumer Confidence per spec §5.2 / §7.4.
    """
    out = df.copy()

    # Pivot top-N table to wide: per station, rank-1..3 distance + counter_id.
    pivoted = top_n_table.pivot(
        index="station_id", columns="counter_rank", values=["counter_id", "distance_km"]
    )
    pivoted.columns = [f"{val}_top{rank}" for val, rank in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Distance columns for ranks 1..3.
    for rank in (1, 2, 3):
        col = f"distance_km_top{rank}"
        if col in pivoted.columns:
            out = out.merge(
                pivoted[["station_id", col]].rename(
                    columns={col: f"ctx_traffic_top{rank}_distance_km"}
                ),
                on="station_id",
                how="left",
            )

    # Lag-N daily counts per (counter_id, date) join. clean.traffic emits
    # the FK column as `station_key`; spatial.nearest emits it as `counter_id`.
    # Normalise both names to `counter_id` for the join.
    traffic = _normalise_traffic_daily(traffic_daily)
    if traffic is not None and not traffic.empty:
        traffic["traffic_lag_1"] = traffic.groupby("counter_id")["daily_total"].shift(1)
        traffic["traffic_lag_7"] = traffic.groupby("counter_id")["daily_total"].shift(7)
        for rank in (1, 2, 3):
            counter_col = f"counter_id_top{rank}"
            if counter_col not in pivoted.columns:
                continue
            station_counter = pivoted[["station_id", counter_col]].rename(
                columns={counter_col: "counter_id"}
            )
            joined = (
                out[["station_id", "date"]]
                .merge(station_counter, on="station_id", how="left")
                .merge(
                    traffic[["counter_id", "date", "traffic_lag_1", "traffic_lag_7"]],
                    on=["counter_id", "date"],
                    how="left",
                )
            )
            out[f"ctx_traffic_top{rank}_lag_1"] = joined["traffic_lag_1"].values
            out[f"ctx_traffic_top{rank}_lag_7"] = joined["traffic_lag_7"].values
    else:
        for rank in (1, 2, 3):
            out[f"ctx_traffic_top{rank}_lag_1"] = pd.NA
            out[f"ctx_traffic_top{rank}_lag_7"] = pd.NA

    # Radius count from summary.
    radius_col = next(
        (c for c in summary_table.columns if c.startswith("stn_n_counters_within_")),
        None,
    )
    if radius_col is not None:
        out = out.merge(
            summary_table[["station_id", radius_col]].rename(
                columns={radius_col: "ctx_traffic_5km_radius_count"}
            ),
            on="station_id",
            how="left",
        )
    else:
        out["ctx_traffic_5km_radius_count"] = pd.NA

    # Apply 50 km cutoff per spec §7.4.
    if "ctx_traffic_top1_distance_km" in out.columns:
        too_far = out["ctx_traffic_top1_distance_km"] > 50.0
        for col in out.columns:
            if col.startswith("ctx_traffic_"):
                out.loc[too_far, col] = pd.NA

    # Phase 5 macro features. Each is None-tolerant: if the upstream
    # parquet wasn't fetched, the column ships as null and LightGBM
    # handles natively.
    out = _add_macro_feature(
        out,
        macro=cash_rate,
        value_col="cash_rate",
        feature_col="ctx_cash_rate",
        lag_days=0,  # forward-fill the latest published value
    )
    out = _add_macro_feature(
        out,
        macro=asx200,
        value_col="close",
        feature_col="ctx_asx200_lag_1",
        lag_days=1,  # yesterday's close
    )
    out = _add_macro_feature(
        out,
        macro=inflation_expectations,
        value_col="inflation_expectations",
        feature_col="ctx_inflation_expectations_lag_7",
        lag_days=7,
    )

    return out


def _add_macro_feature(
    df: pd.DataFrame,
    *,
    macro: pd.DataFrame | None,
    value_col: str,
    feature_col: str,
    lag_days: int,
) -> pd.DataFrame:
    """Forward-fill `macro[value_col]` to daily, lag by `lag_days`, join on date.

    Adds `feature_col` to df (null when macro is None / empty).
    """
    out = df.copy()
    if macro is None or macro.empty or value_col not in macro.columns:
        out[feature_col] = pd.NA
        return out

    macro = macro.assign(date=pd.to_datetime(macro["date"]).dt.date).sort_values("date")
    upstream_start = macro["date"].min()
    panel_end = max(out["date"].max(), macro["date"].max())
    full_dates = pd.date_range(upstream_start, panel_end, freq="D").date
    series = (
        pd.DataFrame({"date": full_dates})
        .merge(macro[["date", value_col]], on="date", how="left")
        .ffill()
    )
    series[feature_col] = series[value_col].shift(lag_days)
    return out.merge(series[["date", feature_col]], on="date", how="left")


# ============================================================
# 7.5 Static station block
# ============================================================


def add_station_features(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Brand columns, competitor counts, terminal distance, metro flag."""
    cols = ["station_id"]
    for c in (
        "brand_raw",
        "brand_canonical",
        "brand_is_major",
        "lat",
        "lon",
        "sa2_name",
    ):
        if c in stations.columns:
            cols.append(c)

    s = stations[cols].copy()
    s = s.rename(
        columns={
            "brand_raw": "stn_brand_raw",
            "brand_canonical": "stn_brand_canonical",
            "brand_is_major": "stn_brand_is_major",
        }
    )

    # stn_is_metro: heuristic from sa2_name for v1 (spec §7.5 amendment).
    if "sa2_name" in s.columns:
        s["stn_is_metro"] = s["sa2_name"].apply(_is_metro_sa2_name)
    else:
        s["stn_is_metro"] = pd.NA

    # Competitor counts via spatial join on station coords.
    competitors = _compute_competitor_counts(stations)
    s = s.merge(competitors, on="station_id", how="left")

    # Stn_is_franchisee: stub null per spec §13 Q3.
    s["stn_is_franchisee"] = pd.NA

    # Drop columns we used for derivation but don't want to expose.
    s = s.drop(columns=[c for c in ("lat", "lon", "sa2_name") if c in s.columns])

    return df.merge(s, on="station_id", how="left")


def _is_metro_sa2_name(sa2_name: object) -> bool:
    """Heuristic: SA2s in greater-Sydney/Newcastle/Wollongong/etc are metro.

    Robust to nulls (returns False).
    """
    if not isinstance(sa2_name, str):
        return False
    return any(prefix in sa2_name for prefix in METRO_SA2_PREFIXES)


def _compute_competitor_counts(stations: pd.DataFrame) -> pd.DataFrame:
    """For each station, count distinct other stations within 2 km and 5 km."""
    if "lat" not in stations.columns or "lon" not in stations.columns:
        return pd.DataFrame(
            {
                "station_id": stations["station_id"],
                "stn_competitors_within_2km": 0,
                "stn_competitors_within_5km": 0,
            }
        )

    s = stations[["station_id", "lat", "lon"]].dropna()
    if s.empty:
        return pd.DataFrame(
            {
                "station_id": stations["station_id"],
                "stn_competitors_within_2km": 0,
                "stn_competitors_within_5km": 0,
            }
        )

    from sklearn.neighbors import BallTree

    rad = np.radians(s[["lat", "lon"]].to_numpy(dtype=np.float64))
    tree = BallTree(rad, metric="haversine")
    earth_km = 6371.0
    n_2km = tree.query_radius(rad, r=2.0 / earth_km, count_only=True) - 1  # exclude self
    n_5km = tree.query_radius(rad, r=5.0 / earth_km, count_only=True) - 1

    by_id = pd.DataFrame(
        {
            "station_id": s["station_id"].to_numpy(),
            "stn_competitors_within_2km": n_2km.astype(np.int64),
            "stn_competitors_within_5km": n_5km.astype(np.int64),
        }
    )
    return stations[["station_id"]].merge(by_id, on="station_id", how="left").fillna(0)


# ============================================================
# 7.6 Weather block
# ============================================================


def add_weather_features(df: pd.DataFrame, weather_dir: Path | None) -> pd.DataFrame:
    """Join per-station weather parquets on `(station_id, date)`.

    Weather is best-effort — if the directory doesn't exist or a station
    has no cached weather file, the wx_* columns are added as nulls and
    LightGBM handles them natively.
    """
    wx_cols = ("wx_temp_max_c", "wx_temp_min_c", "wx_precipitation_mm",
               "wx_wind_speed_max_kmh", "wx_weather_code")

    if weather_dir is None or not weather_dir.exists():
        out = df.copy()
        for col in wx_cols:
            out[col] = pd.NA
        return out

    pieces: list[pd.DataFrame] = []
    for station_id in df["station_id"].unique():
        path = weather_dir / f"{station_id}.parquet"
        if not path.exists():
            continue
        wx = pd.read_parquet(path)
        wx["station_id"] = station_id
        wx["date"] = pd.to_datetime(wx["date"]).dt.date
        pieces.append(wx)

    if not pieces:
        out = df.copy()
        for col in wx_cols:
            out[col] = pd.NA
        return out

    weather = pd.concat(pieces, ignore_index=True)
    keep = ["station_id", "date", *wx_cols]
    weather = weather[[c for c in keep if c in weather.columns]]
    return df.merge(weather, on=["station_id", "date"], how="left")


# ============================================================
# 7.7 Demographic block
# ============================================================

SA2_FEATURE_COLS: tuple[str, ...] = (
    "sa2_total_population",
    "sa2_median_age",
    "sa2_median_household_income_weekly",
    "sa2_seifa_irsd_score",
    "sa2_pct_drive_to_work",
    "sa2_motor_vehicles_per_dwelling",
    "sa2_pct_renters",
    "sa2_pct_employed_full_time",
    "sa2_pct_aged_65_plus",
    "sa2_pct_one_parent_family",
)


def add_sa2_features(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Join the §7.7 demographic block from `stations.parquet`.

    Phase 3 ships 4 of these populated; the other 6 are null stubs
    per spec §7.7.1. This function joins whatever is present.
    """
    cols = ["station_id"] + [c for c in SA2_FEATURE_COLS if c in stations.columns]
    sa2 = stations[cols].copy()
    out = df.merge(sa2, on="station_id", how="left")

    # Add any deferred columns that aren't in stations yet, as nulls.
    for col in SA2_FEATURE_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    return out


# ============================================================
# 7.8 Targets
# ============================================================


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """`y_t1` and `y_t1_t7` shifted within (station_id, 'U91').

    Per spec §3 + §7.8: only U91 rows carry a target. Diesel rows get
    null targets — they exist purely as cross-fuel feature inputs and
    are filtered out at training time.
    """
    out = df.sort_values(["station_id", "fuel_code", "date"]).reset_index(drop=True)
    is_u91 = out["fuel_code"] == "U91"

    # y_t1: tomorrow's price.
    grouped = out.groupby(["station_id", "fuel_code"], sort=False, observed=True)
    out["y_t1"] = grouped["price_mean"].shift(-1)
    # y_t1_t7: mean of price[t+1..t+7]. Compute the past-7 rolling mean
    # then shift up by 7 so row t holds the mean of the next-7-day window.
    out["y_t1_t7"] = grouped["price_mean"].transform(
        lambda s: s.rolling(7, min_periods=7).mean().shift(-7)
    )

    # Null out targets on Diesel rows.
    out.loc[~is_u91, "y_t1"] = pd.NA
    out.loc[~is_u91, "y_t1_t7"] = pd.NA

    return out


# ============================================================
# Orchestrator
# ============================================================


def make_features(
    panel: pd.DataFrame,
    *,
    brent: pd.DataFrame,
    audusd: pd.DataFrame,
    stations: pd.DataFrame,
    top_n: pd.DataFrame,
    summary: pd.DataFrame,
    traffic_daily: pd.DataFrame,
    weather_dir: Path | None = None,
    school_terms_path: Path | None = None,
    aip_tgp: pd.DataFrame | None = None,
    cash_rate: pd.DataFrame | None = None,
    asx200: pd.DataFrame | None = None,
    inflation_expectations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compose all feature blocks. Returns the full features.parquet shape.

    Phase-5 inputs (`aip_tgp`, `cash_rate`, `asx200`,
    `inflation_expectations`) are optional — when omitted, the
    corresponding columns ship as null.
    """
    logger.info("starting feature build: %d panel rows", len(panel))

    df = add_lag_features(panel)
    logger.info("after lag block: %d cols", len(df.columns))

    df = add_upstream_features(df, brent=brent, audusd=audusd, aip_tgp=aip_tgp)
    logger.info("after upstream block: %d cols", len(df.columns))

    df = add_calendar_features(df, school_terms_path=school_terms_path)
    logger.info("after calendar block: %d cols", len(df.columns))

    df = add_context_features(
        df,
        top_n_table=top_n,
        summary_table=summary,
        traffic_daily=traffic_daily,
        cash_rate=cash_rate,
        asx200=asx200,
        inflation_expectations=inflation_expectations,
    )
    logger.info("after ctx block: %d cols", len(df.columns))

    df = add_station_features(df, stations=stations)
    logger.info("after stn block: %d cols", len(df.columns))

    df = add_weather_features(df, weather_dir=weather_dir)
    logger.info("after weather block: %d cols", len(df.columns))

    df = add_sa2_features(df, stations=stations)
    logger.info("after sa2 block: %d cols", len(df.columns))

    df = add_targets(df)
    logger.info("after targets: %d cols, %d rows", len(df.columns), len(df))

    return df


def make_features_from_paths(
    panel_path: Path,
    out_path: Path,
    *,
    brent_path: Path | None = None,
    audusd_path: Path | None = None,
    stations_path: Path | None = None,
    top_n_path: Path | None = None,
    summary_path: Path | None = None,
    traffic_daily_path: Path | None = None,
    weather_dir: Path | None = None,
    school_terms_path: Path | None = None,
    aip_tgp_path: Path | None = None,
    cash_rate_path: Path | None = None,
    asx200_path: Path | None = None,
    inflation_expectations_path: Path | None = None,
) -> None:
    """File-IO convenience wrapper around `make_features`.

    Phase-5 paths are optional — missing files become null feature
    columns rather than fatal errors, so feature builds work
    incrementally as upstream fetchers come online.
    """
    raw = config.DATA_RAW
    interim = config.DATA_INTERIM
    brent_path = brent_path or raw / "brent.parquet"
    audusd_path = audusd_path or raw / "audusd.parquet"
    stations_path = stations_path or interim / "stations.parquet"
    top_n_path = top_n_path or interim / "station_to_counter.parquet"
    summary_path = summary_path or interim / "station_to_counter_summary.parquet"
    traffic_daily_path = traffic_daily_path or interim / "traffic_daily.parquet"
    weather_dir = weather_dir or raw / "weather"
    school_terms_path = school_terms_path or config.DATA_STATIC / "nsw_school_terms.csv"
    aip_tgp_path = aip_tgp_path or raw / "aip_tgp.parquet"
    cash_rate_path = cash_rate_path or raw / "cash_rate.parquet"
    asx200_path = asx200_path or raw / "asx200.parquet"
    inflation_expectations_path = (
        inflation_expectations_path or raw / "inflation_expectations.parquet"
    )

    panel = pd.read_parquet(panel_path)
    brent = pd.read_parquet(brent_path)
    audusd = pd.read_parquet(audusd_path)
    stations = pd.read_parquet(stations_path)
    top_n = pd.read_parquet(top_n_path)
    summary = pd.read_parquet(summary_path)
    traffic_daily = (
        pd.read_parquet(traffic_daily_path) if traffic_daily_path.exists() else pd.DataFrame()
    )
    aip_tgp = pd.read_parquet(aip_tgp_path) if aip_tgp_path.exists() else None
    cash_rate = pd.read_parquet(cash_rate_path) if cash_rate_path.exists() else None
    asx200 = pd.read_parquet(asx200_path) if asx200_path.exists() else None
    inflation_expectations = (
        pd.read_parquet(inflation_expectations_path)
        if inflation_expectations_path.exists()
        else None
    )

    features = make_features(
        panel,
        brent=brent,
        audusd=audusd,
        stations=stations,
        top_n=top_n,
        summary=summary,
        traffic_daily=traffic_daily,
        weather_dir=weather_dir,
        school_terms_path=school_terms_path,
        aip_tgp=aip_tgp,
        cash_rate=cash_rate,
        asx200=asx200,
        inflation_expectations=inflation_expectations,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    logger.info("wrote %d rows x %d cols to %s", len(features), len(features.columns), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--brent", type=Path, default=None)
    parser.add_argument("--audusd", type=Path, default=None)
    parser.add_argument("--stations", type=Path, default=None)
    parser.add_argument("--top-n", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--traffic-daily", type=Path, default=None)
    parser.add_argument("--weather-dir", type=Path, default=None)
    parser.add_argument("--school-terms", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    make_features_from_paths(
        args.panel,
        args.out,
        brent_path=args.brent,
        audusd_path=args.audusd,
        stations_path=args.stations,
        top_n_path=args.top_n,
        summary_path=args.summary,
        traffic_daily_path=args.traffic_daily,
        weather_dir=args.weather_dir,
        school_terms_path=args.school_terms,
    )


if __name__ == "__main__":
    main()
