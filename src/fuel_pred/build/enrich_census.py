"""Enrich stations with SA2 code + Census-derived features + SEIFA.

Reads:
- ``data/interim/stations.parquet`` (post Phase 2 — has lat/lon)
- ``data/raw/seifa_2021_sa2.parquet`` (from fetch.seifa)

Writes (overwrites in place):
- ``data/interim/stations.parquet`` with the §6.1 SA2 columns +
  the §7.7 ``sa2_*`` enrichment block.

Pipeline:
1. Call ``census_augment.Pipeline.augment(df, latitude_column='lat',
   longitude_column='lon')`` — uses pre-resolved coordinates so no G-NAF
   or Nominatim is hit. Augmentor does the SA2 spatial join + variable
   lookup against the GCP DataPack.
2. Join SEIFA on ``sa2_code`` to add ``sa2_seifa_irsd_score``.
3. Stub the 6 deferred derived percentages with nulls (per spec §7.7.1).

Phase 3 v1 ships 4 ``sa2_*`` columns; the 6 derived ratios stay null
until either the EDA notebook surfaces priorities or augmentor #11
ships native derived-variable support. See spec.md §7.7.1.

Spec: spec.md §6.1, §7.7, §12 Phase 3.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fuel_pred import config

logger = logging.getLogger(__name__)

# Direct GCP variables we ship in Phase 3 v1 — keys become the column
# suffixes (augmentor produces e.g. `sa2_total_population` from key
# `total_population`).
DIRECT_VARIABLES: dict[str, str] = {
    "total_population": "G01.Tot_P_P",
    "median_age": "G02.Median_age_persons",
    "median_household_income_weekly": "G02.Median_tot_hhd_inc_weekly",
}

# Spec §7.7 lists 10 sa2_* columns. The 6 below are DERIVED
# (numerator/denominator from GCP tables) — see spec §7.7.1 for why
# they're stubbed null in Phase 3 v1.
DEFERRED_DERIVED_COLUMNS: tuple[str, ...] = (
    "sa2_pct_drive_to_work",
    "sa2_motor_vehicles_per_dwelling",
    "sa2_pct_renters",
    "sa2_pct_employed_full_time",
    "sa2_pct_aged_65_plus",
    "sa2_pct_one_parent_family",
)

# Schema we add to stations.parquet.
ENRICHED_COLUMNS: tuple[str, ...] = (
    "sa2_code",
    "sa2_name",
    "sa2_total_population",
    "sa2_median_age",
    "sa2_median_household_income_weekly",
    "sa2_seifa_irsd_score",
    *DEFERRED_DERIVED_COLUMNS,
)


def _ensure_columns_exist(stations: pd.DataFrame) -> pd.DataFrame:
    """Add expected output columns as nulls so the writer's schema is stable."""
    out = stations.copy()
    for col in ENRICHED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _augment(stations: pd.DataFrame, *, pipeline_factory: object | None = None) -> pd.DataFrame:
    """Call the augmentor's spatial+enrichment path on (lat, lon) pairs.

    `pipeline_factory` is a test seam — when provided, replaces the real
    `census_augment.Pipeline.create(...)` so unit tests don't hit S3.
    """
    if pipeline_factory is None:
        from census_augment import Pipeline

        pipeline = Pipeline.create(
            variables=DIRECT_VARIABLES,
            user_agent=config.USER_AGENT,
            latitude_column="lat",
            longitude_column="lon",
        )
    else:
        pipeline = pipeline_factory()  # type: ignore[operator]

    result = pipeline.augment(  # type: ignore[attr-defined]
        stations, latitude_column="lat", longitude_column="lon"
    )
    return result.df  # type: ignore[no-any-return]


def _join_seifa(stations: pd.DataFrame, seifa_path: Path) -> pd.DataFrame:
    """Add `sa2_seifa_irsd_score` by joining the SEIFA parquet on sa2_code."""
    if not seifa_path.exists():
        logger.warning("SEIFA parquet missing at %s — sa2_seifa_irsd_score stays null", seifa_path)
        return stations

    seifa = pd.read_parquet(seifa_path)[["sa2_code", "irsd_score"]].rename(
        columns={"irsd_score": "sa2_seifa_irsd_score"}
    )
    seifa["sa2_code"] = seifa["sa2_code"].astype(str)

    out = stations.copy()
    if "sa2_code" not in out.columns:
        logger.warning("stations have no sa2_code yet — skipping SEIFA join")
        return out
    out["sa2_code"] = out["sa2_code"].astype(str)

    # If the column already exists from _ensure_columns_exist, drop the
    # null version so the merge brings in fresh values.
    if "sa2_seifa_irsd_score" in out.columns:
        out = out.drop(columns=["sa2_seifa_irsd_score"])
    return out.merge(seifa, on="sa2_code", how="left")


def _check_acceptance(stations: pd.DataFrame, threshold: float = 0.95) -> None:
    """Spec §12 Phase 3: at least `threshold` of stations enriched."""
    n = len(stations)
    if n == 0:
        logger.warning("acceptance check skipped: zero stations")
        return

    columns_to_check = (
        "sa2_code",
        "sa2_total_population",
        "sa2_median_age",
        "sa2_median_household_income_weekly",
        "sa2_seifa_irsd_score",
    )
    for col in columns_to_check:
        if col not in stations.columns:
            logger.warning("acceptance: column %s missing", col)
            continue
        non_null = int(stations[col].notna().sum())
        coverage = non_null / n
        marker = "OK" if coverage >= threshold else "FAIL"
        logger.info(
            "[%s] %s coverage: %.1f%% (%d / %d)", marker, col, 100 * coverage, non_null, n
        )
        if coverage < threshold:
            logger.warning(
                "acceptance threshold (%.0f%%) not met for %s — investigate", threshold * 100, col
            )


def enrich(
    stations_path: Path,
    out_path: Path,
    *,
    seifa_path: Path | None = None,
    force: bool = False,
    pipeline_factory: object | None = None,
) -> None:
    """Add SA2 + Census + SEIFA columns to ``stations.parquet``.

    Args:
        stations_path: input stations parquet (post Phase 2 — needs lat/lon).
        out_path: output parquet (in-place safe; written atomically via .tmp).
        seifa_path: SEIFA parquet (default ``data/raw/seifa_2021_sa2.parquet``).
        force: if True, re-enrich every row even if `sa2_code` is populated.
        pipeline_factory: test seam — replaces the real augmentor pipeline.
    """
    seifa_path = seifa_path or (config.DATA_RAW / "seifa_2021_sa2.parquet")

    stations = pd.read_parquet(stations_path)
    stations = _ensure_columns_exist(stations)
    logger.info("loaded %d stations from %s", len(stations), stations_path)

    if "lat" not in stations.columns or "lon" not in stations.columns:
        raise RuntimeError(
            "stations parquet missing lat/lon — run spatial.resolve_addrs first"
        )

    to_enrich_idx = (
        stations.index if force else stations.index[stations["sa2_code"].isna()]
    )
    logger.info("enriching %d / %d stations", len(to_enrich_idx), len(stations))

    if len(to_enrich_idx) > 0:
        # Subset to rows with lat/lon — augmentor needs both.
        to_enrich = stations.loc[to_enrich_idx]
        usable_mask = to_enrich["lat"].notna() & to_enrich["lon"].notna()
        usable = to_enrich[usable_mask].copy()
        if len(usable) == 0:
            logger.warning("no rows with lat/lon — nothing to enrich")
        else:
            # The augmentor refuses to run if it sees pre-existing columns
            # whose names collide with what it produces (sa2_*, geo_*).
            # Strip them from the input subset, then map results back by
            # positional alignment.
            target_cols = ["sa2_code", "sa2_name"] + [f"sa2_{k}" for k in DIRECT_VARIABLES]
            stripped = usable.drop(
                columns=[c for c in target_cols if c in usable.columns],
                errors="ignore",
            ).reset_index(drop=True)
            original_index = usable.index
            augmented = _augment(stripped, pipeline_factory=pipeline_factory)
            augmented.index = original_index

            # Cast each target column to `object` first — pandas refuses to
            # upcast a float64 column to mixed-type values when an existing
            # typed value is being overwritten.
            for col in target_cols:
                if col in stations.columns:
                    stations[col] = stations[col].astype(object)
            for col in target_cols:
                if col in augmented.columns:
                    stations.loc[original_index, col] = augmented.loc[original_index, col].values

    # SEIFA join always runs against the latest sa2_code values.
    stations = _join_seifa(stations, seifa_path)
    # Re-add stub columns that the merge may have stripped.
    for col in DEFERRED_DERIVED_COLUMNS:
        if col not in stations.columns:
            stations[col] = pd.NA

    _check_acceptance(stations)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    stations.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    tmp.replace(out_path)
    logger.info("wrote %d stations to %s", len(stations), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seifa", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    enrich(args.in_path, args.out, seifa_path=args.seifa, force=args.force)


if __name__ == "__main__":
    main()
