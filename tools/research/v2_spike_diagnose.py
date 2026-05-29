"""Diagnose why GCP temporal returned nulls for 2017 rows.

Dump every column for a single station across both dates; check whether the
2016 GCP DataPack is being read at all (vs e.g. resolving but returning a
DataFrame with column-name drift).
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(r"C:\repos\cauldnz\aus-fuel-forecaster")
STATIONS = REPO_ROOT / "data" / "interim" / "stations.parquet"


def main() -> None:
    from census_augment import Pipeline

    # Single station × 2 dates — simplest possible repro.
    sids = pd.read_parquet(STATIONS, columns=["station_id", "lat", "lon"])
    sids = sids[sids["lat"].notna() & sids["lon"].notna()].head(1)
    s = sids.iloc[0]
    df = pd.DataFrame(
        {
            "station_id": [s["station_id"], s["station_id"]],
            "date": [date(2017, 6, 15), date(2023, 6, 15)],
            "lat": [s["lat"], s["lat"]],
            "lon": [s["lon"], s["lon"]],
        }
    )

    variables = {
        "g01_total_pop": "G01.Tot_P_P",
        "g02_median_age": "G02.Median_age_persons",
        "seifa_irsd": "SEIFA.irsd_score",
    }

    pipeline = Pipeline.create(
        variables=variables,
        user_agent="fuel-pred/0.1 diagnose (chris.auld@auld.nz)",
        latitude_column="lat",
        longitude_column="lon",
        date_column="date",
    )

    result = pipeline.augment(df)
    print("\n=== releases_used ===")
    print(result.releases_used)

    print("\n=== full output (transposed) ===")
    print(result.df.T.to_string())

    # Now try loading the raw GCP fetcher for the 2016 release directly to see
    # what columns it exposes — does G01.Tot_P_P exist for 2016, or did the
    # column name drift?
    print("\n=== GCP 2016 fetcher inspection ===")
    try:
        from census_augment.datasets._gcp import GcpDataSource
        ds = GcpDataSource()
        # Probe both releases if the API supports it
        if hasattr(ds, "list_releases"):
            print("releases:", ds.list_releases())
        # Try loading a 2016-tagged release
        if hasattr(ds, "load"):
            for release in ["2016", "2021"]:
                try:
                    if "release" in ds.load.__code__.co_varnames:
                        loaded = ds.load(release=release)
                    else:
                        loaded = ds.load()
                    print(f"release={release} -> cols (first 30): {list(loaded.columns)[:30]}")
                    print(f"release={release} -> shape: {loaded.shape}")
                    if "Tot_P_P" in loaded.columns:
                        print(f"release={release} -> Tot_P_P exists ✓")
                    else:
                        # Try common variants
                        candidates = [c for c in loaded.columns if "Tot_P" in c or "tot_p" in c.lower()]
                        print(f"release={release} -> Tot_P_P not found; candidates: {candidates[:10]}")
                except Exception as e:
                    print(f"release={release} -> {type(e).__name__}: {e}")
    except ImportError as e:
        print(f"can't import GcpDataSource: {e}")


if __name__ == "__main__":
    main()
