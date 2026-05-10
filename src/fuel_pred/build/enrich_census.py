"""Enrich stations with SA2 code + Census-derived features + SEIFA.

Reads:
- ``data/interim/stations.parquet`` (post Phase 2 — has lat/lon)
- SEIFA: lazy-fetched via the augmentor's ``SeifaDataSource``
  (``data/raw/seifa/`` cache; ABS Census-tied, refreshed every 5 years)

Writes (overwrites in place):
- ``data/interim/stations.parquet`` with the §6.1 SA2 columns +
  the §7.7 ``sa2_*`` enrichment block (now all 10 columns populated).

Pipeline:
1. Call ``census_augment.Pipeline.augment(df, latitude_column='lat',
   longitude_column='lon')`` with the 3 direct GCP variables AND the 6
   PRESETs for the derived percentages. The augmentor (v1.4.0+) treats
   ``PRESET.<id>`` as a first-class variable namespace, auto-loading
   numerator + denominator GCP source columns and evaluating the ratio
   against the right denominator (per
   https://github.com/cauldnz/abs-census-augmentor/issues/11 + #18).
2. Fetch SEIFA via ``census_augment.datasets._seifa.SeifaDataSource``
   and join on ``sa2_code`` to add ``sa2_seifa_irsd_score``.

All 10 spec §7.7 ``sa2_*`` columns are now populated. The previous
DEFERRED_DERIVED_COLUMNS null-stubbing is gone — closed by augmentor
v1.4.2 which fixed the PRESET column refs against the real GCP DataPack
(see https://github.com/cauldnz/abs-census-augmentor/issues/23).

UPSTREAM_GCP_COLLISION: augmentor v1.4.2 has a bug where requesting a
direct GCP variable (e.g. ``G01.Tot_P_P``) AND a PRESET that uses the
same code as a source (e.g. ``PRESET.pct_aged_65_plus``) crashes inside
``_build_gcp_lookup`` with ``ValueError: cannot reindex on an axis with
duplicate labels`` because the dispatch's friendly→code mapping doesn't
dedupe and ``table_df[codes].rename(...)`` collapses duplicates. We work
around it here by splitting colliding requests across two
``Pipeline.augment(...)`` passes and merging column-wise. Issue body
queued in ``tools/upstream_issue_gcp_preset_collision.md`` for the
maintainer to file.

Spec: spec.md §6.1, §7.7, §12 Phase 3.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fuel_pred import config

logger = logging.getLogger(__name__)

# Variables we ask the augmentor to compute on each station's SA2.
# Three direct GCP fields + six PRESETs (curated derived ratios with the
# right denominator pre-baked, augmentor v1.4.0+). Keys here become the
# column suffixes — `total_population` → `sa2_total_population`,
# `pct_renters` → `sa2_pct_renters`, etc.
DIRECT_VARIABLES: dict[str, str] = {
    "total_population": "G01.Tot_P_P",
    "median_age": "G02.Median_age_persons",
    "median_household_income_weekly": "G02.Median_tot_hhd_inc_weekly",
    "pct_drive_to_work": "PRESET.pct_drive_to_work",
    "motor_vehicles_per_dwelling": "PRESET.motor_vehicles_per_dwelling",
    "pct_renters": "PRESET.pct_renters",
    "pct_employed_full_time": "PRESET.pct_employed_full_time",
    "pct_aged_65_plus": "PRESET.pct_aged_65_plus",
    "pct_one_parent_family": "PRESET.pct_one_parent_family",
}

# Schema we add to stations.parquet — all 10 sa2_* columns from
# spec §7.7 are now populated by augmentor v1.4.2+.
ENRICHED_COLUMNS: tuple[str, ...] = (
    "sa2_code",
    "sa2_name",
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


def _ensure_columns_exist(stations: pd.DataFrame) -> pd.DataFrame:
    """Add expected output columns as nulls so the writer's schema is stable."""
    out = stations.copy()
    for col in ENRICHED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _split_for_gcp_collision(variables: dict[str, str]) -> list[dict[str, str]]:
    """Group ``variables`` into augmentor-safe subsets.

    See module docstring (UPSTREAM_GCP_COLLISION). When a direct GCP ref
    (``G\\d+.<col>``) shares its code with the source fields of any
    requested ``PRESET.<id>``, a single ``Pipeline.augment(...)`` call
    crashes. We split the colliding direct refs into a separate group
    so each call sees a non-colliding subset.

    The common (no collision) case returns ``[variables]`` — one group,
    one augment call. Tests rely on this single-call behaviour via
    their stub ``DIRECT_VARIABLES`` fixture.
    """
    try:
        from census_augment.features import features
    except ImportError:  # augmentor not installed (e.g. some test envs)
        return [variables]

    direct_codes: dict[str, str] = {}  # friendly -> "Gxx.<code>"
    preset_ids: list[str] = []
    for friendly, ref in variables.items():
        if ref.startswith("PRESET."):
            preset_ids.append(ref[len("PRESET.") :])
        elif ref and ref[0].upper() == "G" and "." in ref:
            direct_codes[friendly] = ref

    if not direct_codes or not preset_ids:
        return [variables]

    direct_refs = set(direct_codes.values())
    colliding_friendlies: set[str] = set()
    for preset_id in preset_ids:
        try:
            sources = set(features.get(preset_id).source_fields())
        except KeyError:
            continue
        for friendly, ref in direct_codes.items():
            if ref in sources:
                colliding_friendlies.add(friendly)
                # Note: we intentionally don't break — a single direct
                # ref could collide with multiple PRESETs, but the
                # collision is symmetric so one membership flag suffices.
                _ = direct_refs  # silence pyflakes; kept for readability above

    if not colliding_friendlies:
        return [variables]

    pass_a = {f: r for f, r in variables.items() if f not in colliding_friendlies}
    pass_b = {f: r for f, r in variables.items() if f in colliding_friendlies}
    logger.info(
        "augmentor split: %d non-colliding vars + %d colliding direct vars (%s) "
        "to work around upstream #UPSTREAM_GCP_COLLISION",
        len(pass_a),
        len(pass_b),
        sorted(colliding_friendlies),
    )
    return [pass_a, pass_b]


def _augment_one_pass(
    stations: pd.DataFrame,
    variables: dict[str, str],
    *,
    pipeline_factory: object | None = None,
) -> pd.DataFrame:
    """Single ``Pipeline.augment`` call. See ``_augment`` for the multi-pass driver."""
    if pipeline_factory is None:
        from census_augment import Pipeline

        pipeline = Pipeline.create(
            variables=variables,
            user_agent=config.USER_AGENT,
            latitude_column="lat",
            longitude_column="lon",
        )
    else:
        # Test seam: stubs may take no args (legacy) or accept variables (preferred).
        try:
            pipeline = pipeline_factory(variables=variables)  # type: ignore[operator]
        except TypeError:
            pipeline = pipeline_factory()  # type: ignore[operator]

    result = pipeline.augment(  # type: ignore[attr-defined]
        stations, latitude_column="lat", longitude_column="lon"
    )
    return result.df  # type: ignore[no-any-return]


def _augment(stations: pd.DataFrame, *, pipeline_factory: object | None = None) -> pd.DataFrame:
    """Call the augmentor on ``stations`` and return the enriched DataFrame.

    Splits into multiple passes when ``DIRECT_VARIABLES`` triggers the
    UPSTREAM_GCP_COLLISION bug. Merges per-pass ``sa2_*`` blocks
    column-wise; the first pass supplies ``sa2_code``/``sa2_name`` and
    the row scaffold.
    """
    groups = _split_for_gcp_collision(DIRECT_VARIABLES)
    frames = [_augment_one_pass(stations, g, pipeline_factory=pipeline_factory) for g in groups]
    if len(frames) == 1:
        return frames[0]

    merged = frames[0].copy()
    for extra in frames[1:]:
        for col in extra.columns:
            if col.startswith("sa2_") and col not in {"sa2_code", "sa2_name"}:
                merged[col] = extra[col].values
    return merged


def _load_seifa(seifa_cache_dir: Path, seifa_loader: object | None = None) -> pd.DataFrame:
    """Fetch + parse SEIFA via the augmentor's native SeifaDataSource.

    Returns a DataFrame indexed by `sa2_code_2021` (string, 9-digit) with
    one column per index/flavour (irsd_score, irsd_aus_decile, etc).

    `seifa_loader` is a test seam — when provided, it's called instead
    of constructing a real SeifaDataSource. Returns a DataFrame.
    """
    if seifa_loader is not None:
        return seifa_loader()  # type: ignore[operator,no-any-return]

    # SeifaDataSource is exposed under the private `_seifa` submodule in
    # v1.3.0 (the public `census_augment.datasets` namespace ships only
    # `Registry`, `DatasetSpec`, etc — see augmentor #19).
    from census_augment.datasets._seifa import SeifaDataSource

    seifa_cache_dir.mkdir(parents=True, exist_ok=True)
    ds = SeifaDataSource(root=seifa_cache_dir)
    return ds.load()


def _join_seifa(
    stations: pd.DataFrame,
    seifa_cache_dir: Path,
    *,
    seifa_loader: object | None = None,
) -> pd.DataFrame:
    """Add `sa2_seifa_irsd_score` by joining augmentor SEIFA on sa2_code.

    The augmentor's SEIFA frame is indexed by `sa2_code_2021` (string)
    and exposes 46 columns (4 indexes x score+ranks+deciles+percentiles
    + state breakdowns + suppression indicators). For Phase 3 v1 we lift
    only `irsd_score` into `sa2_seifa_irsd_score` per spec §7.7; the
    additional richness lands as a follow-up.
    """
    out = stations.copy()
    if "sa2_code" not in out.columns:
        logger.warning("stations have no sa2_code yet — skipping SEIFA join")
        return out

    try:
        seifa = _load_seifa(seifa_cache_dir, seifa_loader=seifa_loader)
    except Exception as exc:
        logger.warning(
            "SEIFA fetch via augmentor failed (%s: %s) — sa2_seifa_irsd_score stays null",
            type(exc).__name__,
            exc,
        )
        return out

    if "irsd_score" not in seifa.columns:
        logger.warning(
            "augmentor SEIFA frame missing `irsd_score` (cols=%s) — skipping join",
            list(seifa.columns)[:10],
        )
        return out

    seifa_lookup = (
        seifa.reset_index()
        .rename(columns={"sa2_code_2021": "sa2_code", "irsd_score": "sa2_seifa_irsd_score"})
        [["sa2_code", "sa2_seifa_irsd_score"]]
    )
    seifa_lookup["sa2_code"] = seifa_lookup["sa2_code"].astype(str)
    out["sa2_code"] = out["sa2_code"].astype(str)

    # Drop the null stub column added by `_ensure_columns_exist` so the
    # merge brings in fresh values rather than producing _x / _y suffixes.
    if "sa2_seifa_irsd_score" in out.columns:
        out = out.drop(columns=["sa2_seifa_irsd_score"])
    return out.merge(seifa_lookup, on="sa2_code", how="left")


def _check_acceptance(stations: pd.DataFrame, threshold: float = 0.95) -> None:
    """Spec §12 Phase 3: at least `threshold` of stations enriched."""
    n = len(stations)
    if n == 0:
        logger.warning("acceptance check skipped: zero stations")
        return

    # All 10 sa2_* spec §7.7 columns are now populated post augmentor v1.4.2.
    columns_to_check = ENRICHED_COLUMNS
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
    seifa_cache_dir: Path | None = None,
    force: bool = False,
    pipeline_factory: object | None = None,
    seifa_loader: object | None = None,
) -> None:
    """Add SA2 + Census + SEIFA columns to ``stations.parquet``.

    Args:
        stations_path: input stations parquet (post Phase 2 — needs lat/lon).
        out_path: output parquet (in-place safe; written atomically via .tmp).
        seifa_cache_dir: where SeifaDataSource caches the downloaded ABS
            workbook (~150 KB, refreshed on a 5-year Census cycle).
            Defaults to ``data/raw/seifa/``.
        force: if True, re-enrich every row even if `sa2_code` is populated.
        pipeline_factory: test seam — replaces the real augmentor pipeline.
        seifa_loader: test seam — replaces SeifaDataSource construction.
    """
    seifa_cache_dir = seifa_cache_dir or (config.DATA_RAW / "seifa")

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
    stations = _join_seifa(stations, seifa_cache_dir, seifa_loader=seifa_loader)

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
    parser.add_argument(
        "--seifa-cache",
        type=Path,
        default=None,
        help="Cache dir for the augmentor's SeifaDataSource (default: data/raw/seifa)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    enrich(args.in_path, args.out, seifa_cache_dir=args.seifa_cache, force=args.force)


if __name__ == "__main__":
    main()
