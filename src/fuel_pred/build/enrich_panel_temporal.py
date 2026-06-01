"""Per-(station, date) SA2 enrichment via augmentor temporal mode.

Reads:
- ``data/interim/panel.parquet`` (from ``build.panel_grid``) — one row per
  (station_id, fuel_code, date). Fuel is irrelevant to SA2 so we dedupe to
  unique (station_id, date) before sending to the augmentor.
- ``data/interim/stations.parquet`` (post Phase 3) for lat/lon lookup.

Writes:
- ``data/interim/panel_sa2_temporal.parquet`` with one row per unique
  (station_id, date) and the ``sa2_*`` temporal columns from
  ``config.AUGMENTOR_VARIABLES_TEMPORAL`` (SEIFA + DSS + ERP.population_total).

Pipeline:
1. Read the panel; project to (station_id, date) and dedupe.
2. Left-join lat/lon from stations.parquet. Rows without lat/lon are
   logged + dropped — they can't be spatially resolved by the augmentor.
3. Call ``census_augment.Pipeline.augment(df, date_column='date', ...)``
   with ``config.AUGMENTOR_VARIABLES_TEMPORAL``. The augmentor buckets
   rows by (resolved_release, source_edition) before doing one spatial
   join per bucket — so wall-clock scales with #unique-releases, not #rows.
4. Drop the lat/lon + augmentor-emitted metadata (``sa2_code``,
   ``sa2_code_edition``, ``<dataset>_release``, etc.) so the output schema
   is clean: (station_id, date) + ``sa2_*`` value columns.
5. Atomic write to ``panel_sa2_temporal.parquet`` via ``.tmp`` + rename.

Spec: spec.md §7.7.2 (temporal-mode adoption, replaces deferred status).

Coverage notes:
- SEIFA values are dense — every NSW SA2 has 2016 + 2021 scores. Acceptance
  threshold is ≥95% non-null per row, matching the cross-sectional check.
- ERP ``population_total`` historical projection (augmentor #92 fix) works
  back to release year 2017. Pre-2017 panel rows resolve to release 2017
  (no earlier release registered upstream).
- DSS quarterly releases start at 2022-Q4 (the earliest available). Train
  fold rows (≤ 2022-12-31) resolve to 2022-Q4 via ``closest_at_or_before``;
  val (2023) and test (2024-26) rows get per-quarter variation.
- Per-SA2 small-cell suppression on DSS produces legitimate nulls in
  sparse rural SA2s — not gated, only logged per column.

Run via:
    uv run python -m fuel_pred.build.enrich_panel_temporal \\
        --panel data/interim/panel.parquet \\
        --stations data/interim/stations.parquet \\
        --out data/interim/panel_sa2_temporal.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fuel_pred import config

logger = logging.getLogger(__name__)

# Variables we send to the temporal augment pass. Mirrors
# ``config.AUGMENTOR_VARIABLES_TEMPORAL`` directly; kept as a local alias
# for the test seam (same pattern as ``enrich_census.DIRECT_VARIABLES``).
TEMPORAL_VARIABLES: dict[str, str] = dict(config.AUGMENTOR_VARIABLES_TEMPORAL)

# Schema we write out — (station_id, date) keys + one ``sa2_*`` value
# column per TEMPORAL_VARIABLES key.
OUTPUT_VALUE_COLUMNS: tuple[str, ...] = tuple(f"sa2_{key}" for key in TEMPORAL_VARIABLES)
OUTPUT_COLUMNS: tuple[str, ...] = ("station_id", "date", *OUTPUT_VALUE_COLUMNS)


def _load_unique_panel_keys(
    panel_path: Path,
    stations_path: Path,
) -> pd.DataFrame:
    """Project the panel to unique (station_id, date) + join lat/lon."""
    panel = pd.read_parquet(panel_path, columns=["station_id", "date"])
    n_panel_rows = len(panel)
    keys = panel.drop_duplicates(subset=["station_id", "date"]).reset_index(drop=True)
    logger.info(
        "panel: %d rows → %d unique (station_id, date) tuples",
        n_panel_rows,
        len(keys),
    )

    stations = pd.read_parquet(stations_path, columns=["station_id", "lat", "lon"])
    stations["station_id"] = stations["station_id"].astype(str)
    keys["station_id"] = keys["station_id"].astype(str)
    keys = keys.merge(stations, on="station_id", how="left")

    n_missing_latlon = int(keys["lat"].isna().sum() + keys["lon"].isna().sum())
    if n_missing_latlon:
        before = len(keys)
        keys = keys[keys["lat"].notna() & keys["lon"].notna()].reset_index(drop=True)
        logger.warning(
            "dropped %d rows missing lat/lon (%d → %d)",
            before - len(keys),
            before,
            len(keys),
        )
    return keys


def _build_temporal_pipeline(variables: dict[str, str]):
    """Construct a temporal-mode Pipeline with ``out_of_range='nearest'``.

    Extracted so each pass of the collision splitter can build its own
    Pipeline against a smaller variable subset. Real-augmentor path only;
    the test seam in ``_augment`` short-circuits this.
    """
    from census_augment import Pipeline
    from census_augment.config import (
        Config,
        DataSourcesConfig,
        GeocodingConfig,
        InputConfig,
        NominatimConfig,
        OutputConfig,
        TemporalConfig,
    )

    # Pipeline.create() doesn't expose temporal config; build Config
    # manually so we can set out_of_range='nearest'.
    cfg = Config(
        input=InputConfig(
            latitude_column="lat",
            longitude_column="lon",
            date_column="date",
        ),
        output=OutputConfig(prefix="sa2_"),
        geocoding=GeocodingConfig(
            providers=["nominatim"],
            nominatim=NominatimConfig(user_agent=config.USER_AGENT),
        ),
        data_sources=DataSourcesConfig(),
        variables=variables,
        temporal=TemporalConfig(out_of_range="nearest"),
    )
    return Pipeline.from_config(cfg)


def _augment_one_pass(
    keys: pd.DataFrame,
    variables: dict[str, str],
    *,
    pipeline_factory: object | None = None,
):
    """Single ``Pipeline.augment`` call in temporal mode. See ``_augment``
    for the multi-pass driver (collision-splitter aware).
    """
    if pipeline_factory is None:
        pipeline = _build_temporal_pipeline(variables)
    else:
        # Test seam: stubs may take no args (legacy) or accept variables.
        try:
            pipeline = pipeline_factory(variables=variables)  # type: ignore[operator]
        except TypeError:
            pipeline = pipeline_factory()  # type: ignore[operator]

    result = pipeline.augment(keys)  # type: ignore[attr-defined]
    logger.info(
        "  temporal pass complete: %d rows × %d cols (releases: %s)",
        len(result.df),  # type: ignore[attr-defined]
        len(result.df.columns),  # type: ignore[attr-defined]
        getattr(result, "releases_used", {}),
    )
    return result.df  # type: ignore[no-any-return]


def _augment(
    keys: pd.DataFrame,
    *,
    pipeline_factory: object | None = None,
) -> pd.DataFrame:
    """Run the temporal-mode augment, splitting on PRESET collisions.

    Returns the merged augmented frame with the augmentor's full output
    schema; callers project to ``OUTPUT_COLUMNS`` separately.

    Splits into multiple passes when ``TEMPORAL_VARIABLES`` triggers the
    same PRESET-collision bug as the cross-sectional pass (see
    ``build._augmentor_helpers``). Merges per-pass ``sa2_*`` blocks
    column-wise; the first pass supplies bookkeeping
    (``sa2_code``/``sa2_name``/``*_release``) and the row scaffold.

    Temporal config:
    - ``out_of_range='nearest'`` — pre-earliest-release rows clamp to
      the earliest available release (e.g. pre-2022-Q4 train rows
      get the 2022-Q4 DSS values, same as cross-sectional). The
      augmentor logs a WARNING per affected row.
    - ``resolution='closest_at_or_before'`` (default) — causally safe;
      no peek-ahead at quarter midpoints.
    """
    from fuel_pred.build._augmentor_helpers import (
        merge_augmented_frames,
        split_for_preset_collision,
    )

    groups = split_for_preset_collision(TEMPORAL_VARIABLES)
    logger.info(
        "temporal augment: %d unique (station_id, date) rows, %d variables in %d pass(es)",
        len(keys),
        len(TEMPORAL_VARIABLES),
        len(groups),
    )
    frames = [
        _augment_one_pass(keys, g, pipeline_factory=pipeline_factory) for g in groups
    ]
    # Temporal mode adds *_release / sa2_code_edition / *_sa2_code_source
    # bookkeeping cols per dataset family. The merge helper must not
    # overwrite the first pass's metadata when subsequent passes carry
    # their own; pass them as primary_key_cols.
    bookkeeping = tuple(
        c
        for c in frames[0].columns
        if c.endswith("_release")
        or c.endswith("_sa2_code_source")
        or c in {"sa2_code_edition", "sa2_resolution"}
    )
    return merge_augmented_frames(frames, primary_key_cols=bookkeeping)


def _check_acceptance(out: pd.DataFrame, threshold: float = 0.95) -> None:
    """Log per-column coverage. SEIFA gated strictly; DSS/ERP advisory only."""
    n = len(out)
    if n == 0:
        logger.warning("acceptance check skipped: zero rows")
        return
    seifa_cols = [c for c in OUTPUT_VALUE_COLUMNS if c.startswith("sa2_seifa_")]
    other_cols = [c for c in OUTPUT_VALUE_COLUMNS if c not in seifa_cols]
    for col in seifa_cols:
        non_null = int(out[col].notna().sum())
        coverage = non_null / n
        marker = "OK" if coverage >= threshold else "FAIL"
        logger.info(
            "[%s] %s coverage: %.1f%% (%d / %d)",
            marker,
            col,
            100 * coverage,
            non_null,
            n,
        )
        if coverage < threshold:
            logger.warning(
                "acceptance threshold (%.0f%%) not met for %s — investigate",
                threshold * 100,
                col,
            )
    for col in other_cols:
        non_null = int(out[col].notna().sum())
        coverage = non_null / n
        logger.info(
            "[advisory] %s coverage: %.1f%% (%d / %d)",
            col,
            100 * coverage,
            non_null,
            n,
        )


def enrich(
    panel_path: Path,
    stations_path: Path,
    out_path: Path,
    *,
    pipeline_factory: object | None = None,
) -> None:
    """Build ``panel_sa2_temporal.parquet`` from a panel + stations input.

    Args:
        panel_path: input panel parquet (one row per (station_id, fuel_code,
            date) — we dedupe internally to unique (station_id, date)).
        stations_path: input stations parquet for lat/lon.
        out_path: output parquet (atomic via .tmp + rename).
        pipeline_factory: test seam — replaces the real augmentor pipeline
            with a stub that returns a deterministic DataFrame. The factory
            takes either no args or a ``variables`` kw.
    """
    keys = _load_unique_panel_keys(panel_path, stations_path)
    if len(keys) == 0:
        raise RuntimeError(
            f"no usable rows after dedup + lat/lon filter against "
            f"{panel_path} + {stations_path}"
        )

    augmented = _augment(keys, pipeline_factory=pipeline_factory)

    # Project to (station_id, date, sa2_<value cols>). Drop the augmentor's
    # bookkeeping columns (sa2_code, sa2_code_edition, <dataset>_release,
    # etc.) — they're useful for debugging but not modelled.
    missing = [c for c in OUTPUT_VALUE_COLUMNS if c not in augmented.columns]
    if missing:
        raise RuntimeError(
            f"augmentor output missing expected columns: {missing}. "
            f"Got columns: {sorted(augmented.columns)}"
        )
    out = augmented[["station_id", "date", *OUTPUT_VALUE_COLUMNS]].copy()

    _check_acceptance(out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    tmp.replace(out_path)
    logger.info("wrote %d rows × %d cols to %s", len(out), len(out.columns), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--stations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    enrich(args.panel, args.stations, args.out)


if __name__ == "__main__":
    main()
