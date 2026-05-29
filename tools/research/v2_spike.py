"""Research spike: census-augmentor v2.0.0 in temporal mode.

Goal: verify that v2.0's per-row release selection actually produces *different*
SA2 features for old vs. new fuel-panel rows at the same station.

Why this matters: today we join SA2 once (via stations.parquet enriched with
2021 Census values) and broadcast that single row across 2016-2026 dates. If
v2.0 temporal mode just falls back to "use 2021 for everything" (because no
older Census release covers the variable), the upgrade buys us nothing on the
DSS/SEIFA/GCP front. If it actually swaps in 2016 values for pre-2021 dates,
spec §7.7.2 (temporal-DSS) becomes a real phase worth planning.

Strategy:
- Sample 25 rows from 2017 + 25 rows from 2023 at the SAME set of stations.
- Join lat/lon from stations.parquet.
- Run Pipeline.augment with date_column='date' against v2.0.0.
- Print per-row sa2_population, sa2_release columns, and any *_release suffixes.
- Tabulate which datasets actually changed snapshot between 2017 and 2023.

This script does NOT modify the project venv. It uses `uv run --no-project
--with` to ephemerally install v2.0.0 in a one-shot subprocess.

Usage (from project root):
    uv run --no-project \\
        --with "abs-census-augmentor @ git+https://github.com/cauldnz/abs-census-augmentor.git@v2.0.0" \\
        --with pandas \\
        --with pyarrow \\
        python tools/research/v2_spike.py

Output: prints to stdout. Capture with `> tools/research/v2_spike.out.txt`.

Cache: census-augment writes to ~/.cache/census-augment/ — first run will
download ~hundreds of MB (ASGS Ed.2 + Ed.3 boundaries, GCP DataPacks 2016+2021,
SEIFA 2016+2021, ERP). Re-runs are fast. The cache is independent of the
project's data/raw/ tree and survives reruns.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(r"C:\repos\cauldnz\aus-fuel-forecaster")
PANEL = REPO_ROOT / "data" / "interim" / "panel.parquet"
STATIONS = REPO_ROOT / "data" / "interim" / "stations.parquet"

# Sample dates that straddle the 2016 vs 2021 Census line, and the
# 2016 vs 2021 SEIFA line. ERP releases annually, so we expect different
# `erp_by_sa2_release` between these two too.
# Panel stores `date` as Python datetime.date objects (not Timestamps).
DATE_2017 = date(2017, 6, 15)
DATE_2023 = date(2023, 6, 15)

N_STATIONS = 25


def build_sample() -> pd.DataFrame:
    """Sample 25 stations that have rows on both DATE_2017 and DATE_2023."""
    panel = pd.read_parquet(PANEL, columns=["station_id", "date"])
    stations = pd.read_parquet(STATIONS, columns=["station_id", "lat", "lon"])

    in_2017 = set(
        panel.loc[panel["date"] == DATE_2017, "station_id"].astype(str)
    )
    in_2023 = set(
        panel.loc[panel["date"] == DATE_2023, "station_id"].astype(str)
    )
    both = in_2017 & in_2023
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations[stations["lat"].notna() & stations["lon"].notna()]
    candidates = stations[stations["station_id"].isin(both)]

    # Deterministic sample (no Math.random/Date.now usage to satisfy
    # workflow-style determinism; we just pick the first N alphabetically).
    chosen = candidates.sort_values("station_id").head(N_STATIONS).copy()
    print(f"Candidate stations with both 2017+2023 rows: {len(candidates)}")
    print(f"Chosen sample: {len(chosen)} stations × 2 dates = {len(chosen) * 2} rows")

    rows = []
    for _, r in chosen.iterrows():
        for d in (DATE_2017, DATE_2023):
            rows.append(
                {
                    "station_id": r["station_id"],
                    "date": d,
                    "lat": r["lat"],
                    "lon": r["lon"],
                }
            )
    return pd.DataFrame(rows)


def run_temporal_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Run v2.0 augmentor with date_column set."""
    from census_augment import Pipeline  # noqa: WPS433 (intentional late import)

    # GCP + SEIFA only — the two datasets that now register both 2016 and 2021
    # releases. ERP is intentionally excluded: testing revealed that the
    # ERP release index only enumerates the most recent annual publication
    # (e.g. '2024'), so temporal mode raises RuntimeError on 2017 rows.
    # ABS publishes ERP as one annual workbook containing the full time
    # series back to 2001 — there isn't a separate '2017' release to load.
    # DSS PRESETs are excluded for the same reason (would require ERP
    # denominator and inherit the same limitation).
    variables = {
        "total_population": "G01.Tot_P_P",
        "median_age": "G02.Median_age_persons",
        "median_income": "G02.Median_tot_hhd_inc_weekly",
        "seifa_irsd": "SEIFA.irsd_score",
    }

    pipeline = Pipeline.create(
        variables=variables,
        user_agent="fuel-pred/0.1 spike (chris.auld@auld.nz)",
        latitude_column="lat",
        longitude_column="lon",
        date_column="date",
    )

    print("\n=== Running Pipeline.augment in TEMPORAL mode ===")
    print(f"Variables requested: {list(variables.keys())}")
    print(f"Input rows: {len(df)}")
    result = pipeline.augment(df)

    print("\n=== releases_used (per-dataset) ===")
    for ds, releases in result.releases_used.items():
        print(f"  {ds}: {releases}")

    return result.df


def compare(out: pd.DataFrame) -> None:
    """Diff the per-station 2017 vs 2023 rows."""
    print("\n=== Per-row release columns (first 6 stations) ===")
    release_cols = [c for c in out.columns if c.endswith("_release")]
    show = ["station_id", "date", "sa2_code"] + release_cols
    show = [c for c in show if c in out.columns]
    print(out.head(12)[show].to_string(index=False))

    print("\n=== Cross-edition codes (if any) ===")
    edition_cols = [c for c in out.columns if "edition" in c or "source" in c]
    if edition_cols:
        print(out.head(12)[["station_id", "date"] + edition_cols].to_string(index=False))
    else:
        print("(no *_edition / *_source columns in output)")

    print("\n=== Per-station 2017 vs 2023 value deltas ===")
    value_cols = [
        "sa2_total_population",
        "sa2_median_age",
        "sa2_median_income",
        "sa2_seifa_irsd",
    ]
    value_cols = [c for c in value_cols if c in out.columns]
    if not value_cols:
        print("(no expected sa2_* value columns found)")
        print("All output columns:", list(out.columns))
        return

    # `date` is a Python date object — pivot directly on it; we know it
    # only takes two values (DATE_2017 / DATE_2023) so use it as columns.
    out["year"] = out["date"].apply(lambda d: d.year)
    wide = out.pivot_table(
        index="station_id",
        columns="year",
        values=value_cols,
        aggfunc="first",
    )

    print(f"\nValue columns inspected: {value_cols}")
    print("Stations × (var × year) pivot (first 6 stations):")
    print(wide.head(6).to_string())

    # Summarize which variables actually differ between the two years
    print("\n=== Summary: how often does each variable differ between 2017 and 2023? ===")
    for col in value_cols:
        try:
            y_old = wide[(col, 2017)]
            y_new = wide[(col, 2023)]
        except KeyError:
            print(f"  {col}: missing one year")
            continue
        # Float-equality tolerant; ignore both-null rows
        both_present = y_old.notna() & y_new.notna()
        if both_present.sum() == 0:
            print(f"  {col}: no overlap")
            continue
        diff = (y_old[both_present] - y_new[both_present]).abs()
        n_diff = int((diff > 1e-9).sum())
        n_eq = int((diff <= 1e-9).sum())
        median_old = float(y_old[both_present].median())
        median_new = float(y_new[both_present].median())
        print(
            f"  {col}: {n_diff}/{n_diff + n_eq} stations differ — "
            f"median 2017={median_old:.2f}, 2023={median_new:.2f}"
        )


def main() -> None:
    print("=== census-augmentor v2.0.0 temporal-mode spike ===")
    print(f"Python: {sys.version.split()[0]}")
    try:
        import census_augment
        print(f"census_augment: {census_augment.__version__ if hasattr(census_augment, '__version__') else '(no __version__)'}")
    except ImportError as e:
        print(f"FATAL: census_augment not importable: {e}")
        sys.exit(2)

    df = build_sample()
    out = run_temporal_mode(df)
    compare(out)
    print("\n=== Spike complete ===")


if __name__ == "__main__":
    main()
