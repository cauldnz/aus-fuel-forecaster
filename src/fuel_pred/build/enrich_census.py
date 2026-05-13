"""Enrich stations with SA2 code + Census-derived features + SEIFA + ERP + DSS + ABS_PIA.

Reads:
- ``data/interim/stations.parquet`` (post Phase 2 — has lat/lon).
- The augmentor's local caches under ``data/raw/`` (one subdir per dataset:
  ``boundaries/<edition>/``, ``census/<edition>/``, ``mb/<edition>/``,
  ``seifa_2021/``, ``erp_by_sa2/``, ``dss_payments/``, ``abs_personal_income/``).
  Per-edition layout is required by augmentor v1.5+ (Temporal Phase D).

Writes (overwrites in place):
- ``data/interim/stations.parquet`` with the §6.1 SA2 keys + the §7.7
  ``sa2_*`` enrichment block (28 columns at v1.5).

Pipeline:
1. One ``Pipeline.augment(stations, latitude_column='lat', longitude_column='lon')``
   call requesting the full ``config.AUGMENTOR_VARIABLES`` set. The augmentor
   v1.5 routes each ``<NAMESPACE>.<field>`` reference to the right dataset
   (GCP, SEIFA, ERP, DSS, ABS_PIA) and returns one DataFrame with all columns
   prefixed ``sa2_``. PRESET refs are evaluated through the same call.
2. The result is written out.

The previous bespoke SEIFA path (``SeifaDataSource`` direct call + post-augment
join) is retired: v1.5 dispatches ``SEIFA.<field>`` through the same registry
as the other namespaces, so we don't need a parallel code path.

UPSTREAM_GCP_COLLISION (workaround retained pending verification): augmentor
v1.4.2 had a bug where requesting a direct GCP variable (e.g. ``G01.Tot_P_P``)
AND a PRESET that uses the same code as a source (e.g. ``PRESET.pct_aged_65_plus``)
crashed inside ``_build_gcp_lookup`` with ``ValueError: cannot reindex on an
axis with duplicate labels``. We work around it by splitting colliding requests
across two ``Pipeline.augment(...)`` passes and merging column-wise. v1.5 may
have fixed this — verify with an integration test before removing the splitter.
Tracking comment: ``tools/upstream_issue_gcp_preset_collision.md``.

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
# Sourced from ``config.AUGMENTOR_VARIABLES`` so the spec / config file is the
# single source of truth — see that module for per-namespace commentary.
DIRECT_VARIABLES: dict[str, str] = dict(config.AUGMENTOR_VARIABLES)

# Schema we add to stations.parquet — sa2_code / sa2_name come from the
# augmentor's geographic resolution; the rest are 1:1 with DIRECT_VARIABLES
# keys (each prefixed ``sa2_`` per the augmentor's ``output_prefix`` default).
ENRICHED_COLUMNS: tuple[str, ...] = (
    "sa2_code",
    "sa2_name",
    *(f"sa2_{key}" for key in DIRECT_VARIABLES),
)

# Subset that gets the strict ≥95% acceptance check. ERP / ABS_PIA / DSS
# columns are excluded because they have legitimate per-SA2 nulls (publication
# coverage gaps + DSS small-cell suppression — see spec §7.7). We log per-column
# coverage for them but don't fail the run.
ACCEPTANCE_PREFIXES: tuple[str, ...] = (
    "sa2_code",
    "sa2_name",
    "sa2_total_population",
    "sa2_median_age",
    "sa2_median_household_income_weekly",
    "sa2_pct_",
    "sa2_motor_vehicles_per_dwelling",
    "sa2_seifa_",
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
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Single ``Pipeline.augment`` call. See ``_augment`` for the multi-pass driver."""
    if pipeline_factory is None:
        from census_augment import Pipeline

        pipeline = Pipeline.create(
            variables=variables,
            user_agent=config.USER_AGENT,
            latitude_column="lat",
            longitude_column="lon",
            data_dir=data_dir,
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


def _augment(
    stations: pd.DataFrame,
    *,
    pipeline_factory: object | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Call the augmentor on ``stations`` and return the enriched DataFrame.

    Splits into multiple passes when ``DIRECT_VARIABLES`` triggers the
    UPSTREAM_GCP_COLLISION bug. Merges per-pass ``sa2_*`` blocks
    column-wise; the first pass supplies ``sa2_code``/``sa2_name`` and
    the row scaffold.
    """
    groups = _split_for_gcp_collision(DIRECT_VARIABLES)
    frames = [
        _augment_one_pass(
            stations, g, pipeline_factory=pipeline_factory, data_dir=data_dir
        )
        for g in groups
    ]
    if len(frames) == 1:
        return frames[0]

    merged = frames[0].copy()
    for extra in frames[1:]:
        for col in extra.columns:
            if col.startswith("sa2_") and col not in {"sa2_code", "sa2_name"}:
                merged[col] = extra[col].values
    return merged


def _check_acceptance(stations: pd.DataFrame, threshold: float = 0.95) -> None:
    """Spec §12 Phase 3: at least `threshold` of stations enriched.

    Strict (warn-on-fail) check applies only to the dense GCP / SEIFA
    columns enumerated by ``ACCEPTANCE_PREFIXES``. ERP / ABS_PIA / DSS
    columns are logged at INFO without a coverage gate — they have
    legitimate per-SA2 nulls (small-cell suppression, coverage gaps),
    so a < 95% rate isn't necessarily a regression.
    """
    n = len(stations)
    if n == 0:
        logger.warning("acceptance check skipped: zero stations")
        return

    for col in ENRICHED_COLUMNS:
        if col not in stations.columns:
            logger.warning("acceptance: column %s missing", col)
            continue
        non_null = int(stations[col].notna().sum())
        coverage = non_null / n
        is_strict = any(col.startswith(p) for p in ACCEPTANCE_PREFIXES)
        marker = "OK" if (not is_strict or coverage >= threshold) else "FAIL"
        logger.info(
            "[%s] %s coverage: %.1f%% (%d / %d)", marker, col, 100 * coverage, non_null, n
        )
        if is_strict and coverage < threshold:
            logger.warning(
                "acceptance threshold (%.0f%%) not met for %s — investigate",
                threshold * 100, col,
            )


def enrich(
    stations_path: Path,
    out_path: Path,
    *,
    data_dir: Path | None = None,
    force: bool = False,
    pipeline_factory: object | None = None,
) -> None:
    """Add SA2 + Census + SEIFA + ERP + DSS + ABS_PIA columns to ``stations.parquet``.

    Args:
        stations_path: input stations parquet (post Phase 2 — needs lat/lon).
        out_path: output parquet (in-place safe; written atomically via .tmp).
        data_dir: where the augmentor stores per-dataset caches. Defaults to
            ``config.DATA_RAW`` so caches live under ``data/raw/`` alongside
            our other raw inputs (each dataset gets its own subdir per the
            v1.5 layout: ``seifa_2021/``, ``erp_by_sa2/``, ``dss_payments/``,
            ``abs_personal_income/``, plus ``boundaries/<edition>/`` etc.).
        force: if True, re-enrich every row even if `sa2_code` is populated.
        pipeline_factory: test seam — replaces the real augmentor pipeline.
    """
    data_dir = data_dir or config.DATA_RAW

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
            augmented = _augment(
                stripped, pipeline_factory=pipeline_factory, data_dir=data_dir
            )
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
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Augmentor cache root (defaults to data/raw). Each dataset writes "
            "to its own subdir under here per the v1.5 layout."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    enrich(args.in_path, args.out, data_dir=args.data_dir, force=args.force)


if __name__ == "__main__":
    main()
