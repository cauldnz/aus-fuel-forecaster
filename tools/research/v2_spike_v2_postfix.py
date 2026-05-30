"""Verify augmentor #91 + #92 fixes against the latest upstream main.

Three runs:

1. **GCP cross-edition** (#91 Stage 1) — should now raise a loud
   ValueError with concrete workarounds, NOT silently return NaN.

2. **ERP `population_total` temporal** (#92 fix) — should now succeed
   for historical years via column projection. Per-row release matches.

3. **ERP age/sex temporal** (documented limitation from #92 resolution)
   — should return null for historical rows because the source 3235.0
   cube only ships the latest year's demographics.

Plus a smoke pass for SEIFA temporal (should still work; baseline).

Run via:
    uv run --no-project \\
        --with "abs-census-augmentor @ git+https://github.com/cauldnz/abs-census-augmentor.git@main" \\
        --with pandas --with pyarrow \\
        python tools/research/v2_spike_v2_postfix.py

(uses @main, not @v2.0.0, to pick up the post-v2.0.0 fixes that haven't
been tagged yet.)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import traceback

import pandas as pd

REPO_ROOT = Path(r"C:\repos\cauldnz\aus-fuel-forecaster")
STATIONS = REPO_ROOT / "data" / "interim" / "stations.parquet"


def _sample_df(n_stations: int = 3) -> pd.DataFrame:
    """Pick a small deterministic sample with rows on both 2017 and 2023."""
    sids = pd.read_parquet(STATIONS, columns=["station_id", "lat", "lon"])
    sids = sids[sids["lat"].notna() & sids["lon"].notna()].sort_values("station_id").head(n_stations)
    rows = []
    for _, r in sids.iterrows():
        for d in (date(2017, 6, 15), date(2023, 6, 15)):
            rows.append(
                {
                    "station_id": r["station_id"],
                    "date": d,
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                }
            )
    return pd.DataFrame(rows)


def run_and_summarise(label: str, variables: dict[str, str]) -> None:
    from census_augment import Pipeline

    df = _sample_df(3)
    print(f"\n\n{'=' * 70}")
    print(f"=== {label}")
    print(f"=== Variables: {list(variables.keys())}")
    print(f"=== Input: {len(df)} rows (3 stations × 2 dates)")
    print("=" * 70)

    try:
        pipeline = Pipeline.create(
            variables=variables,
            user_agent="fuel-pred/0.1 spike-v2-postfix (chris.auld@auld.nz)",
            latitude_column="lat",
            longitude_column="lon",
            date_column="date",
        )
        result = pipeline.augment(df)
        print(f"\nreleases_used: {result.releases_used}")
        cols_to_show = ["station_id", "date"] + [
            c for c in result.df.columns
            if c.endswith("_release") or c.startswith("sa2_")
        ]
        cols_to_show = [c for c in cols_to_show if c in result.df.columns]
        print(f"\nOutput (first 6 rows, key columns):")
        print(result.df[cols_to_show].head(6).to_string(index=False))

        # For each value column, show null fraction
        value_cols = [
            c for c in result.df.columns
            if c.startswith("sa2_") and not c.endswith(("_release", "_source"))
            and c not in {"sa2_code", "sa2_name", "sa2_code_edition", "sa2_resolution"}
        ]
        print(f"\nNull fractions:")
        for c in value_cols:
            n_null = int(result.df[c].isna().sum())
            print(f"  {c}: {n_null}/{len(result.df)} null")
    except Exception as e:
        print(f"\n{type(e).__name__}: {e}")
        if "ValueError" in type(e).__name__ or "RuntimeError" in type(e).__name__:
            print("(Expected loud-error path)")
        else:
            print("\nFull traceback:")
            traceback.print_exc()


def main() -> None:
    import sys
    print(f"Python: {sys.version.split()[0]}")
    try:
        import census_augment
        ver = getattr(census_augment, "__version__", "(no __version__)")
        print(f"census_augment: {ver}")
    except ImportError as e:
        print(f"FATAL: {e}")
        sys.exit(2)

    # 1. SEIFA-only temporal — should work; baseline for the others
    run_and_summarise(
        "1. SEIFA-only temporal (baseline — should still work)",
        {"seifa_irsd": "SEIFA.irsd_score"},
    )

    # 2. GCP temporal — #91 Stage 1 — should raise loud error for the 2017 rows
    run_and_summarise(
        "2. GCP temporal (#91 Stage 1 — should raise loud error)",
        {
            "seifa_irsd": "SEIFA.irsd_score",
            "g01_total_pop": "G01.Tot_P_P",
        },
    )

    # 3. ERP population_total temporal — #92 — should now succeed for 2017
    run_and_summarise(
        "3. ERP population_total temporal (#92 — should succeed via column projection)",
        {
            "seifa_irsd": "SEIFA.irsd_score",
            "erp_total": "ERP.population_total",
        },
    )

    # 4. ERP age/sex temporal — documented limitation — should return null for 2017
    run_and_summarise(
        "4. ERP age/sex temporal (#92 documented limitation — null for historical)",
        {
            "seifa_irsd": "SEIFA.irsd_score",
            "erp_65_plus": "ERP.population_65_plus",
            "erp_median_age": "ERP.median_age",
        },
    )


if __name__ == "__main__":
    main()
