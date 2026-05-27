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

# Imported here (not inline at point-of-use) to keep ruff E402 clean.
# SA2_FEATURE_COLS is the single source of truth for the SA2 block; importing
# it from feature_blocks keeps make_features in lockstep with the model code —
# when the SA2 block changes, this module picks up the new column list without
# a separate edit.
from fuel_pred.train.feature_blocks import SA2_COLUMNS as SA2_FEATURE_COLS

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
        out.loc[dl_mask, col] = np.nan

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
            # np.nan (not pd.NA) so the column is float64 not object —
            # LightGBM handles NaN-on-float natively, but rejects object.
            daily[col] = np.nan

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

    # Pre-long-weekend Friday flag (spec §13.6 Phase 1). Friday before a
    # Monday public holiday: cal_day_of_week == 4 AND
    # cal_days_to_next_public_holiday == 3 (today=Fri, Sat=1, Sun=2,
    # Mon=3). Derivation is a pure function of the two columns just
    # written, so dtype matches other cal_is_* boolean flags.
    out["cal_is_pre_long_weekend"] = (
        (out["cal_day_of_week"] == 4) & (out["cal_days_to_next_public_holiday"] == 3)
    ).astype("boolean")

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
            out[f"ctx_traffic_top{rank}_lag_1"] = np.nan
            out[f"ctx_traffic_top{rank}_lag_7"] = np.nan

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
        out["ctx_traffic_5km_radius_count"] = np.nan

    # Apply 50 km cutoff per spec §7.4.
    if "ctx_traffic_top1_distance_km" in out.columns:
        too_far = out["ctx_traffic_top1_distance_km"] > 50.0
        for col in out.columns:
            if col.startswith("ctx_traffic_"):
                out.loc[too_far, col] = np.nan

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
        out[feature_col] = np.nan
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


def add_station_features(
    df: pd.DataFrame,
    stations: pd.DataFrame,
    stations_venues_path: Path | None = None,
) -> pd.DataFrame:
    """Brand columns, competitor counts, terminal distance, metro flag, venues.

    Args:
        df: panel rows with at least ``station_id``.
        stations: roster from ``data/interim/stations.parquet``.
        stations_venues_path: optional path to a parquet from
            ``spatial.venues``. When None or the file doesn't exist,
            the 5 venue columns are added as nulls / zero — same
            None-tolerant pattern used for the Tier-2 macros in §7.4.
    """
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
        s["stn_is_metro"] = np.nan

    # Competitor counts via spatial join on station coords.
    competitors = _compute_competitor_counts(stations)
    s = s.merge(competitors, on="station_id", how="left")

    # Stn_is_franchisee: stub null per spec §13 Q3.
    # np.nan → float64 (not pd.NA → object) so LightGBM accepts the
    # column as a numeric feature with all-missing values rather than
    # raising "pandas dtypes must be int, float or bool".
    s["stn_is_franchisee"] = np.nan

    # Venue features (spec §13.6 Phase 1). Optional — when the parquet
    # isn't on disk, attach nulls so the columns are always in the
    # output schema and LightGBM handles them natively.
    s = _attach_venue_features(s, stations_venues_path)

    # Drop columns we used for derivation but don't want to expose.
    s = s.drop(columns=[c for c in ("lat", "lon", "sa2_name") if c in s.columns])

    return df.merge(s, on="station_id", how="left")


# Names of the venue feature columns added by ``_attach_venue_features``;
# kept in lockstep with feature_blocks.VENUE_COLUMNS station-side names.
VENUE_STATION_COLUMNS: tuple[str, ...] = (
    "stn_nearest_venue_km",
    "stn_nearest_venue_capacity",
    "stn_nearest_venue_type",
    "stn_n_venues_within_5km",
)


def _attach_venue_features(
    s: pd.DataFrame, stations_venues_path: Path | None
) -> pd.DataFrame:
    """Merge venue features onto the per-station block; null-fill if absent."""
    if stations_venues_path is None or not stations_venues_path.exists():
        if stations_venues_path is not None:
            logger.info(
                "stations_venues parquet not found at %s — venue columns will be null",
                stations_venues_path,
            )
        for col in VENUE_STATION_COLUMNS:
            if col == "stn_nearest_venue_type":
                # Match the parquet schema (object) so the column doesn't
                # collide with a categorical dtype downstream.
                s[col] = pd.Series([pd.NA] * len(s), dtype="object")
            else:
                s[col] = np.nan
        return s

    venues = pd.read_parquet(stations_venues_path)
    # Only keep the columns the model actually consumes; drop the
    # nearest-venue-id (high-cardinality identifier, redundant with the
    # capacity + type pair).
    keep = ["station_id", *VENUE_STATION_COLUMNS]
    available = [c for c in keep if c in venues.columns]
    missing = [c for c in keep if c not in venues.columns]
    if missing:
        logger.warning(
            "stations_venues missing %d expected column(s): %s — filled with nulls",
            len(missing),
            missing,
        )
    merged = s.merge(venues[available], on="station_id", how="left")
    for col in missing:
        if col == "stn_nearest_venue_type":
            merged[col] = pd.Series([pd.NA] * len(merged), dtype="object")
        else:
            merged[col] = np.nan
    logger.info(
        "merged venue features for %d stations (%d had a nearest-venue match)",
        len(merged),
        int(merged["stn_nearest_venue_km"].notna().sum()),
    )
    return merged


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
    """Join per-station weather parquets, leakage-corrected per spec §13.7.

    Each cached parquet stores ``date`` as the *valid date* of the weather
    observation/forecast. To deliver the day-ahead forecast to a panel row
    at date ``t`` (whose target is price at ``t+1``), we shift the weather
    ``date`` back by 1 day before merging — so the row representing
    weather valid on ``d`` joins onto the panel row at ``d - 1``.

    After the shift, ``wx_*`` columns on the panel row at ``t`` carry the
    day-ahead NWP forecast for ``t+1`` issued on ``t`` (post-2017, via the
    Open-Meteo Historical Forecast API), or the ERA5 persistence proxy
    for 2016 dates (a single, documented residual). See
    ``docs/research/2026-05_weather_leakage_fix.md`` for the full rationale.

    Weather is best-effort — if the directory doesn't exist or a station
    has no cached weather file, the wx_* columns are added as nulls and
    LightGBM handles them natively.
    """
    wx_cols = ("wx_temp_max_c", "wx_temp_min_c", "wx_precipitation_mm",
               "wx_wind_speed_max_kmh", "wx_weather_code")

    if weather_dir is None or not weather_dir.exists():
        out = df.copy()
        for col in wx_cols:
            out[col] = np.nan
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
            out[col] = np.nan
        return out

    weather = pd.concat(pieces, ignore_index=True)
    keep = ["station_id", "date", *wx_cols]
    weather = weather[[c for c in keep if c in weather.columns]]
    # Spec §13.7 (v2.0 leakage fix): shift the weather valid-date back by
    # one day so panel row `t` receives the day-ahead forecast for `t+1`.
    # See docs/research/2026-05_weather_leakage_fix.md §"Pipeline changes".
    # `date` is a Series of python `dt.date` (object dtype); promote to
    # datetime for vectorised arithmetic, then drop back to dt.date so the
    # merge dtype matches the panel's `date` column.
    weather["date"] = (
        pd.to_datetime(weather["date"]) - pd.Timedelta(days=1)
    ).dt.date
    return df.merge(weather, on=["station_id", "date"], how="left")


# ------------------------------------------------------------------
# 7.6b Weather block — NOAA GFS grid-cell join, multi-horizon wide schema
# (spec §13.7 v2.0 + §13.8 v2.1)
# ------------------------------------------------------------------

# Variable bases (matching the per-(date, horizon) parquet's column names
# produced by `fetch.gfs.fetch_and_write_one_day`, minus the `wx_weather_code`
# null-stub which is added here for schema parity with the Open-Meteo path).
_GFS_VALUE_BASES: tuple[str, ...] = (
    "wx_temp_max_c",
    "wx_temp_min_c",
    "wx_precipitation_mm",
    "wx_wind_speed_max_kmh",
)
# wx_weather_code: GFS/GEFS doesn't emit WMO codes (research doc R3).
# Materialised as a null stub for schema consistency with the Open-Meteo path.
_GFS_NULL_BASES: tuple[str, ...] = ("wx_weather_code",)


def _bilinear_weight_columns_for_resolution(resolution_prefix: str) -> dict[str, str]:
    """Map the canonical bilinear column suffixes to the resolution-prefixed
    column names in `station_grid_mapping.parquet`.

    For `resolution_prefix="gfs"` returns e.g.
        {"bl_lat_idx_0": "gfs_bl_lat_idx_0", ..., "bl_w_11": "gfs_bl_w_11"}.
    """
    suffixes = (
        "bl_lat_idx_0", "bl_lat_idx_1",
        "bl_lon_idx_0", "bl_lon_idx_1",
        "bl_w_00", "bl_w_01", "bl_w_10", "bl_w_11",
    )
    return {sfx: f"{resolution_prefix}_{sfx}" for sfx in suffixes}


def _resolution_prefix_for_date(date: dt.date) -> str:
    """Pick the station_grid_mapping column-prefix matching the gfs.py
    date-routing convention (`_select_resolution_for_date`).

    Mirrors the date windows from `fetch.gfs` without importing it (keeps
    `build.make_features` decoupled from the fetcher's module-level guards).
    """
    # Boundaries duplicated from src/fuel_pred/fetch/gfs.py — kept in
    # lockstep there. (Pre-2017 dates: caller must filter; this routes them
    # to gefs1 silently, but no grid parquet will exist so the join nulls.)
    if date <= dt.date(2020, 9, 22):
        return "gefs1"
    if date <= dt.date(2021, 3, 31):
        return "gefs05"
    return "gfs"


def _bilinear_interp_grid_to_stations(
    grid_df: pd.DataFrame,
    mapping_subset: pd.DataFrame,
    resolution_prefix: str,
    value_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Interpolate per-grid-cell values to per-station using the 4-neighbour
    bilinear weights from the station grid mapping.

    Args:
        grid_df: long-form NSW-box grid with columns
            `lat_idx, lon_idx, wx_*`.
        mapping_subset: per-station rows from `station_grid_mapping.parquet`,
            already filtered to one (station_id) row per station relevant
            to the panel.
        resolution_prefix: one of "gfs", "gefs05", "gefs1".
        value_columns: which value columns of grid_df to interpolate.

    Returns DataFrame with columns `station_id, <value_columns...>`.
    """
    suffixes = _bilinear_weight_columns_for_resolution(resolution_prefix)
    # grid_df lookup: (lat_idx, lon_idx) -> dict of value per column.
    # Use a Series indexed on the (lat_idx, lon_idx) pair for fast .get().
    idx = pd.MultiIndex.from_arrays(
        [grid_df["lat_idx"].astype(np.int64), grid_df["lon_idx"].astype(np.int64)],
        names=["lat_idx", "lon_idx"],
    )
    value_frame = grid_df[list(value_columns)].copy()
    value_frame.index = idx

    # The 4 corner index pairs per station.
    corner_specs = (
        ("bl_lat_idx_0", "bl_lon_idx_0", "bl_w_00"),
        ("bl_lat_idx_0", "bl_lon_idx_1", "bl_w_01"),
        ("bl_lat_idx_1", "bl_lon_idx_0", "bl_w_10"),
        ("bl_lat_idx_1", "bl_lon_idx_1", "bl_w_11"),
    )

    n_stations = len(mapping_subset)
    accum = {col: np.zeros(n_stations, dtype=np.float64) for col in value_columns}
    any_corner_present = np.zeros(n_stations, dtype=bool)

    for lat_key, lon_key, w_key in corner_specs:
        lat_indices = mapping_subset[suffixes[lat_key]].to_numpy(dtype=np.int64)
        lon_indices = mapping_subset[suffixes[lon_key]].to_numpy(dtype=np.int64)
        weights = mapping_subset[suffixes[w_key]].to_numpy(dtype=np.float64)

        # Build the per-row corner index, look up each value column once.
        corner_idx = pd.MultiIndex.from_arrays(
            [lat_indices, lon_indices], names=["lat_idx", "lon_idx"],
        )
        # reindex returns NaN for corners outside the NSW grid (shouldn't
        # happen since the mapping rejects out-of-NSW stations upstream,
        # but be defensive).
        corner_values = value_frame.reindex(corner_idx)
        present_mask = corner_values.notna().any(axis=1).to_numpy()
        any_corner_present |= present_mask

        for col in value_columns:
            v = corner_values[col].to_numpy(dtype=np.float64)
            # Where the corner is missing, contribute zero (the weight
            # at that corner is effectively dropped, biasing the interp
            # toward the present corners — acceptable for edge cases).
            v = np.where(np.isnan(v), 0.0, v)
            accum[col] += weights * v

    # Stations with no present corners → all-NaN row (not zero).
    for col in value_columns:
        accum[col] = np.where(any_corner_present, accum[col], np.nan)

    out = pd.DataFrame(
        {"station_id": mapping_subset["station_id"].to_numpy()},
    )
    for col in value_columns:
        out[col] = accum[col]
    return out


def add_weather_features_gfs(
    df: pd.DataFrame,
    weather_gfs_dir: Path | None,
    station_grid_mapping_path: Path | None,
    *,
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
) -> pd.DataFrame:
    """Join NSW GFS grid weather to panel, multi-horizon wide schema.

    File-naming convention (per ``fetch.gfs._grid_parquet_path`` +
    ``fetch_one_day_all_horizons``): the per-file ``date`` is the forecast
    *issue date* (init date), and horizon N's grid covers leads 6..24 hours
    out from `init_date + (N-1) days`. So `<YYYY-MM-DD>_h<N>.parquet`
    contains the forecast issued on YYYY-MM-DD and *valid on*
    `YYYY-MM-DD + N days`.

    For panel row at date ``t`` (target = price at ``t+1``), the day-ahead
    forecast is in `<t>_h1.parquet` — issue date equals the panel date and
    no shift is needed. Multi-horizon: `<t>_h<N>.parquet` carries the
    forecast for `t + N`, materialised as `wx_*_tN` columns on the panel
    row at `t`.

    For each (panel_date, horizon) pair, this function reads the grid
    parquet at ``weather_gfs_dir/<panel_date>_h<horizon>.parquet``, then
    bilinearly interpolates to each panel station using the 4-neighbour
    weights pre-computed in ``station_grid_mapping_path``.

    Wide-schema output: 5 vars × 7 horizons = 35 wx_*_tN columns added.
    ``wx_weather_code_tN`` is a null stub (GFS/GEFS doesn't emit WMO codes;
    see research doc R3). v2.0 (1-day-ahead) model consumes only `_t1`;
    v2.1 (7-day) reuses the same matrix.

    None-tolerant: if ``weather_gfs_dir`` is None / missing, or if
    ``station_grid_mapping_path`` is None / missing, returns df with all
    35 wx_*_tN columns as null (matching the existing
    ``add_weather_features`` pattern).

    Args:
        df: panel DataFrame with at least `station_id` and `date` columns.
        weather_gfs_dir: directory containing per-(date, horizon) grid
            parquets named `<YYYY-MM-DD>_h<N>.parquet`.
        station_grid_mapping_path: path to the parquet produced by
            ``spatial.gfs_grid.compute_station_grid_mapping``.
        horizons: which horizons to materialise. Default 1..7.

    Spec: spec.md §13.7 (v2.0), §13.8 (v2.1).
    """
    all_wx_cols: list[str] = []
    for h in horizons:
        for base in _GFS_VALUE_BASES:
            all_wx_cols.append(f"{base}_t{h}")
        for base in _GFS_NULL_BASES:
            all_wx_cols.append(f"{base}_t{h}")

    # Fast None-tolerant exit — keeps the schema stable when the GFS
    # cache hasn't been built yet.
    if (
        weather_gfs_dir is None
        or not weather_gfs_dir.exists()
        or station_grid_mapping_path is None
        or not station_grid_mapping_path.exists()
    ):
        if weather_gfs_dir is not None and not weather_gfs_dir.exists():
            logger.info(
                "weather_gfs_dir not found at %s — wx_*_tN columns will be null",
                weather_gfs_dir,
            )
        if (
            station_grid_mapping_path is not None
            and not station_grid_mapping_path.exists()
        ):
            logger.info(
                "station_grid_mapping not found at %s — wx_*_tN columns will be null",
                station_grid_mapping_path,
            )
        out = df.copy()
        for col in all_wx_cols:
            out[col] = np.nan
        return out

    mapping = pd.read_parquet(station_grid_mapping_path)
    # Restrict to stations present in the panel — saves work for narrow
    # panels (single station, etc.).
    panel_station_ids = set(df["station_id"].unique())
    mapping = mapping[mapping["station_id"].astype(str).isin(panel_station_ids)].copy()
    mapping["station_id"] = mapping["station_id"].astype(str)

    unique_dates = sorted(set(df["date"].unique()))

    # For each horizon, build a per-(station_id, panel_date) DataFrame of
    # interpolated values; merge each into the panel at the end.
    out = df.copy()

    # Cache loaded grid parquets across horizons (no overlap by default,
    # but the structure is harmless).
    grid_cache: dict[Path, pd.DataFrame | None] = {}

    for horizon in horizons:
        # Renamed value columns for this horizon.
        rename_map = {base: f"{base}_t{horizon}" for base in _GFS_VALUE_BASES}

        pieces: list[pd.DataFrame] = []
        for panel_date in unique_dates:
            grid_path = (
                weather_gfs_dir / f"{panel_date.isoformat()}_h{horizon}.parquet"
            )
            if grid_path not in grid_cache:
                grid_cache[grid_path] = (
                    pd.read_parquet(grid_path) if grid_path.exists() else None
                )
            grid_df = grid_cache[grid_path]
            if grid_df is None:
                continue

            res_prefix = _resolution_prefix_for_date(panel_date)
            station_values = _bilinear_interp_grid_to_stations(
                grid_df=grid_df,
                mapping_subset=mapping,
                resolution_prefix=res_prefix,
                value_columns=_GFS_VALUE_BASES,
            )
            station_values["date"] = panel_date
            pieces.append(station_values)

        if pieces:
            horizon_frame = pd.concat(pieces, ignore_index=True).rename(
                columns=rename_map,
            )
            out = out.merge(horizon_frame, on=["station_id", "date"], how="left")
        else:
            for new_col in rename_map.values():
                out[new_col] = np.nan

        # Null-stub columns for this horizon (e.g. wx_weather_code_tN).
        for base in _GFS_NULL_BASES:
            out[f"{base}_t{horizon}"] = np.nan

    logger.info(
        "add_weather_features_gfs: wrote %d wx_*_tN columns (%d horizons × %d vars)",
        len(all_wx_cols), len(horizons), len(_GFS_VALUE_BASES) + len(_GFS_NULL_BASES),
    )
    return out


# ============================================================
# 7.7 Demographic block
# ============================================================


def add_sa2_features(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Join the §7.7 demographic block from `stations.parquet`.

    The SA2 columns are sourced from ``feature_blocks.SA2_COLUMNS`` so this
    function stays in lockstep with what the model code expects. Columns
    in the block that don't exist in ``stations`` yet are added as nulls
    (then either populated downstream or filtered out by the model's
    feature-list logic).
    """
    cols = ["station_id"] + [c for c in SA2_FEATURE_COLS if c in stations.columns]
    sa2 = stations[cols].copy()
    out = df.merge(sa2, on="station_id", how="left")

    # Add any deferred columns that aren't in stations yet, as nulls.
    # np.nan → float64 (not pd.NA → object) per the convention
    # documented in the upstream / ctx / stn / wx blocks above.
    for col in SA2_FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan

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
    out.loc[~is_u91, "y_t1"] = np.nan
    out.loc[~is_u91, "y_t1_t7"] = np.nan

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
    stations_venues_path: Path | None = None,
    weather_gfs_dir: Path | None = None,
    station_grid_mapping_path: Path | None = None,
) -> pd.DataFrame:
    """Compose all feature blocks. Returns the full features.parquet shape.

    Phase-5 inputs (`aip_tgp`, `cash_rate`, `asx200`,
    `inflation_expectations`) are optional — when omitted, the
    corresponding columns ship as null. ``stations_venues_path`` (spec
    §13.6 Phase 1) is likewise optional: when None or missing, the
    venue columns ship as null.

    GFS weather source (spec §13.7 v2.0 + §13.8 v2.1): if BOTH
    ``weather_gfs_dir`` and ``station_grid_mapping_path`` are provided
    AND exist, the multi-horizon wide-schema GFS path
    (``add_weather_features_gfs``) is used instead of the per-station
    Open-Meteo path. Otherwise falls back to ``add_weather_features``
    (Open-Meteo). Session 4 will add the formal ``WEATHER_SOURCE``
    config switch; for now the routing is purely None-tolerant.
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

    df = add_station_features(
        df, stations=stations, stations_venues_path=stations_venues_path
    )
    logger.info("after stn block: %d cols", len(df.columns))

    # Weather: GFS multi-horizon takes precedence if both inputs supplied
    # AND exist; else Open-Meteo. Session 4 will add the formal config
    # switch. (Existence checks live in the called functions so callers
    # can pass paths that don't exist yet — graceful-null fallback.)
    use_gfs = (
        weather_gfs_dir is not None
        and weather_gfs_dir.exists()
        and station_grid_mapping_path is not None
        and station_grid_mapping_path.exists()
    )
    if use_gfs:
        df = add_weather_features_gfs(
            df,
            weather_gfs_dir=weather_gfs_dir,
            station_grid_mapping_path=station_grid_mapping_path,
        )
    else:
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
    stations_venues_path: Path | None = None,
    weather_gfs_dir: Path | None = None,
    station_grid_mapping_path: Path | None = None,
) -> None:
    """File-IO convenience wrapper around `make_features`.

    Phase-5 paths are optional — missing files become null feature
    columns rather than fatal errors, so feature builds work
    incrementally as upstream fetchers come online. Same applies to
    ``stations_venues_path`` (spec §13.6 Phase 1).
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
    stations_venues_path = stations_venues_path or config.INTERIM_STATIONS_VENUES

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
    # `stations_venues_path` is passed through to `add_station_features`
    # (which lazy-loads); pass None when the file doesn't exist so the
    # function takes the graceful-null branch.
    venues_path = stations_venues_path if stations_venues_path.exists() else None

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
        stations_venues_path=venues_path,
        weather_gfs_dir=weather_gfs_dir,
        station_grid_mapping_path=station_grid_mapping_path,
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
    parser.add_argument(
        "--stations-venues",
        type=Path,
        default=None,
        help=(
            "optional path to data/interim/stations_venues.parquet "
            "(from spatial.venues). When omitted, defaults to "
            "config.INTERIM_STATIONS_VENUES; null feature columns "
            "when the file doesn't exist."
        ),
    )
    parser.add_argument(
        "--weather-gfs-dir",
        type=Path,
        default=None,
        help=(
            "optional dir of per-(date, horizon) NOAA GFS grid parquets "
            "from fetch.gfs. When present alongside --station-grid-mapping, "
            "the multi-horizon wide-schema GFS path is used instead of "
            "the per-station Open-Meteo path (spec §13.7 v2.0 / §13.8 v2.1)."
        ),
    )
    parser.add_argument(
        "--station-grid-mapping",
        type=Path,
        default=None,
        help=(
            "optional path to data/interim/station_grid_mapping.parquet "
            "(from spatial.gfs_grid). Required when --weather-gfs-dir is set."
        ),
    )
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
        stations_venues_path=args.stations_venues,
        weather_gfs_dir=args.weather_gfs_dir,
        station_grid_mapping_path=args.station_grid_mapping,
    )


if __name__ == "__main__":
    main()
