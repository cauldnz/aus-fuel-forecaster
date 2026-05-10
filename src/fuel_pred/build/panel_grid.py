"""Build the dense (station_id, fuel_code, date) panel with forward-fill.

Per spec §6.2, `fuel_daily.parquet` is unbalanced — rows exist only for
days a station submitted a price for that fuel. The feature builder
needs a dense grid so lag / rolling computations don't have to
special-case missing days.

Output (`data/interim/panel.parquet`):

    station_id, fuel_code, date,
    price_mean, price_min, price_max, n_obs

Construction:
  1. Per-station date range = `[first_seen, last_seen]` from
     `stations.parquet` (clamped to the project span).
  2. Cartesian product with `FUELS_V1 = ('U91', 'DL')` per spec — both
     fuels are persisted because U91 features include cross-fuel lags
     against Diesel at the same station (spec §7.1).
  3. Left-join `fuel_daily` on `(station_id, fuel_code, date)`.
  4. Forward-fill `price_mean / price_min / price_max` within
     `(station_id, fuel_code)` up to `max_forward_fill_days` (default 7
     per spec §6.2).
  5. `n_obs` is filled with 0 on inserted rows (no observation that day).

Spec: spec.md §6.2.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fuel_pred import config

logger = logging.getLogger(__name__)

# Per spec §6.2.
DEFAULT_MAX_FORWARD_FILL_DAYS: int = 7

# Output schema.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "station_id",
    "fuel_code",
    "date",
    "price_mean",
    "price_min",
    "price_max",
    "n_obs",
)


def _per_station_date_range(
    stations: pd.DataFrame, span_start: str, span_end: str | None
) -> pd.DataFrame:
    """For each station, build a (station_id, date) frame for its lifetime.

    Each station's range is `[max(first_seen, span_start),
    min(last_seen, span_end)]`. Stations without first_seen/last_seen
    fall back to the project span.
    """
    span_start_d = pd.to_datetime(span_start).date()
    span_end_d = pd.to_datetime(span_end).date() if span_end else None

    pieces: list[pd.DataFrame] = []
    for row in stations.itertuples(index=False):
        # Stations missing first/last_seen entirely (e.g. degenerate data)
        # get the whole project span; better than dropping them silently.
        first = getattr(row, "first_seen", None)
        last = getattr(row, "last_seen", None)
        if pd.notna(first):
            first_d = max(span_start_d, pd.to_datetime(first).date())
        else:
            first_d = span_start_d
        last_d = pd.to_datetime(last).date() if pd.notna(last) else None
        if span_end_d is not None and last_d is not None:
            last_d = min(span_end_d, last_d)
        elif span_end_d is not None:
            last_d = span_end_d
        elif last_d is None:
            # No last_seen and no span_end — skip; can't bound.
            continue

        if last_d < first_d:
            continue
        dates = pd.date_range(first_d, last_d, freq="D").date
        pieces.append(pd.DataFrame({"station_id": row.station_id, "date": dates}))

    if not pieces:
        return pd.DataFrame({"station_id": [], "date": []})
    return pd.concat(pieces, ignore_index=True)


def _cross_with_fuels(station_dates: pd.DataFrame, fuels: tuple[str, ...]) -> pd.DataFrame:
    """Cartesian-multiply the (station_id, date) frame by fuels."""
    if station_dates.empty:
        return pd.DataFrame({"station_id": [], "fuel_code": [], "date": []})
    fuel_df = pd.DataFrame({"fuel_code": list(fuels)})
    fuel_df["_join"] = 1
    sd = station_dates.assign(_join=1)
    out = sd.merge(fuel_df, on="_join").drop(columns="_join")
    return out[["station_id", "fuel_code", "date"]]


def _forward_fill_with_horizon(
    df: pd.DataFrame, group_cols: list[str], value_cols: list[str], horizon_days: int
) -> pd.DataFrame:
    """Forward-fill `value_cols` within `group_cols`, but only across gaps
    of at most `horizon_days` consecutive missing days.

    Returns a copy.
    """
    if df.empty:
        return df.copy()

    out = df.sort_values([*group_cols, "date"]).copy()

    # `g_count` increments each time a new non-null value appears within
    # the group; rows with the same g_count form a "run" we can forward-fill.
    # We also need a "days since last observation in this run" measure to
    # null out values that are >horizon stale.
    notna_marker = out[value_cols[0]].notna()
    out["_seen"] = notna_marker.groupby([out[c] for c in group_cols]).cumsum()

    # Forward-fill values within each group.
    for col in value_cols:
        out[col] = out.groupby(group_cols, sort=False)[col].ffill()

    # Compute "days since last observation" within each group. The last
    # observed date is the max(date) where the original price was non-null,
    # tracked per-row via cummax on the run.
    last_obs_date = out["date"].where(notna_marker)
    last_obs_date = last_obs_date.groupby([out[c] for c in group_cols]).ffill()
    days_since = (pd.to_datetime(out["date"]) - pd.to_datetime(last_obs_date)).dt.days

    # Null out filled values where the gap exceeds the horizon.
    too_stale = days_since > horizon_days
    for col in value_cols:
        out.loc[too_stale, col] = pd.NA

    out = out.drop(columns=["_seen"])
    return out


def build(
    stations: pd.DataFrame,
    fuel_daily: pd.DataFrame,
    *,
    fuels: tuple[str, ...] = config.FUELS_V1,
    span_start: str = config.SPAN_START,
    span_end: str | None = None,
    max_forward_fill_days: int = DEFAULT_MAX_FORWARD_FILL_DAYS,
) -> pd.DataFrame:
    """Build the dense panel from cleaned stations + fuel_daily.

    Args:
        stations: must have `station_id, first_seen, last_seen`.
        fuel_daily: must have `station_id, fuel_code, date, price_mean,
            price_min, price_max, n_obs`.
        fuels: fuels to include in the panel. Spec ships both U91 + DL
            even though the target is U91-only.
        span_start: earliest date to include.
        span_end: latest date to include. Defaults to None → use each
            station's `last_seen`.
        max_forward_fill_days: per spec §6.2, default 7. After this gap
            length, prices revert to null and LightGBM handles it.

    Returns:
        DataFrame with `OUTPUT_COLUMNS` schema. Sorted by
        `(station_id, fuel_code, date)`.
    """
    if "station_id" not in stations.columns:
        raise ValueError("stations must have a 'station_id' column")
    if not fuel_daily.empty:
        required = {"station_id", "fuel_code", "date", "price_mean", "n_obs"}
        missing = required - set(fuel_daily.columns)
        if missing:
            raise ValueError(f"fuel_daily missing columns {missing}")

    station_dates = _per_station_date_range(stations, span_start, span_end)
    grid = _cross_with_fuels(station_dates, fuels)
    logger.info(
        "panel grid before fuel join: %d rows (%d stations x %d fuels)",
        len(grid),
        stations["station_id"].nunique(),
        len(fuels),
    )

    # Coerce date types so merge keys align — fuel_daily's `date` may be
    # datetime64 from an earlier write.
    grid["date"] = pd.to_datetime(grid["date"]).dt.date

    if fuel_daily.empty:
        # Nothing to merge — every row in the grid stays null.
        merged = grid.copy()
        for col in ("price_mean", "price_min", "price_max"):
            merged[col] = pd.NA
        merged["n_obs"] = 0
    else:
        fd = fuel_daily.copy()
        fd["date"] = pd.to_datetime(fd["date"]).dt.date
        # Filter fuel_daily to the requested fuels (defensive).
        fd = fd[fd["fuel_code"].isin(fuels)].copy()
        merged = grid.merge(
            fd[
                ["station_id", "fuel_code", "date", "price_mean", "price_min", "price_max", "n_obs"]
            ],
            on=["station_id", "fuel_code", "date"],
            how="left",
        )

    # Insert n_obs=0 for inserted (gap) rows; price columns stay null
    # until the forward-fill below decides.
    merged["n_obs"] = merged["n_obs"].fillna(0).astype("int64")

    # Forward-fill price columns within (station_id, fuel_code).
    filled = _forward_fill_with_horizon(
        merged,
        group_cols=["station_id", "fuel_code"],
        value_cols=["price_mean", "price_min", "price_max"],
        horizon_days=max_forward_fill_days,
    )

    return filled.loc[:, list(OUTPUT_COLUMNS)].reset_index(drop=True)


def build_from_paths(
    stations_path: Path,
    fuel_daily_path: Path,
    out_path: Path,
    *,
    fuels: tuple[str, ...] = config.FUELS_V1,
    span_start: str = config.SPAN_START,
    span_end: str | None = None,
    max_forward_fill_days: int = DEFAULT_MAX_FORWARD_FILL_DAYS,
) -> None:
    """Read inputs, build panel, write parquet."""
    stations = pd.read_parquet(stations_path)
    fuel_daily = pd.read_parquet(fuel_daily_path)
    logger.info(
        "loaded %d stations + %d fuel-daily rows", len(stations), len(fuel_daily)
    )

    panel = build(
        stations,
        fuel_daily,
        fuels=fuels,
        span_start=span_start,
        span_end=span_end,
        max_forward_fill_days=max_forward_fill_days,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)

    n_with_price = int(panel["price_mean"].notna().sum())
    logger.info(
        "wrote %d panel rows (%d with non-null price_mean) to %s",
        len(panel),
        n_with_price,
        out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", required=True, type=Path)
    parser.add_argument("--fuel", required=True, type=Path,
                        help="Path to fuel_daily.parquet")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--span-start", default=config.SPAN_START)
    parser.add_argument("--span-end", default=None)
    parser.add_argument("--max-forward-fill-days", type=int, default=DEFAULT_MAX_FORWARD_FILL_DAYS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    build_from_paths(
        args.stations,
        args.fuel,
        args.out,
        span_start=args.span_start,
        span_end=args.span_end,
        max_forward_fill_days=args.max_forward_fill_days,
    )


if __name__ == "__main__":
    main()
