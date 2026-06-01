# `compute_sa2_areas_km2` raises `AttributeError: 'NoneType'` on boundaries with null geometry

## Summary

After PR #97 (`feat(erp): add population_density_per_km2 column`),
`Pipeline.create()` / `Pipeline.from_config()` unconditionally calls
`spatial.compute_sa2_areas_km2(loaded_boundaries, ...)` for each
loaded ASGS edition (pipeline.py:430). That function assumes every
row's `geometry` is non-null:

```python
return {
    str(code): float(geom.area / 1_000_000.0)
    for code, geom in zip(in_equal_area[code_column], in_equal_area.geometry, strict=False)
}
```

ASGS boundary GeoDataFrames can carry rows with null `geometry`
(typically `Migratory - Offshore - Shipping` SA2s and a handful of
legitimately-empty pseudo-areas). On any such boundary set, the loop
raises:

```
AttributeError: 'NoneType' object has no attribute 'area'
```

The error blocks **every** `Pipeline.create()` call regardless of which
variables are requested — the area computation runs at pipeline init
time so the bug can't be sidestepped by asking only for non-density
variables.

## Reproducer

Against `main` at `762a6a0f`:

```python
from census_augment import Pipeline

p = Pipeline.create(
    variables={"pop": "G01.Tot_P_P"},
    user_agent="repro/0.1 (you@example.com)",
    latitude_column="lat",
    longitude_column="lon",
)
# Raises:
# AttributeError: 'NoneType' object has no attribute 'area'
```

Stack trace:

```
File ".../census_augment/pipeline.py", line 717, in create
    return cls.from_config(cfg, data_dir=data_dir, cache_dir=cache_dir)
File ".../census_augment/pipeline.py", line 430, in from_config
    sa2_areas_km2 = _compute_areas(loaded_boundaries, code_column=edition.sa2_code_column)
File ".../census_augment/spatial.py", line 180, in compute_sa2_areas_km2
    return {
File ".../census_augment/spatial.py", line 181, in <dictcomp>
    str(code): float(geom.area / 1_000_000.0)
AttributeError: 'NoneType' object has no attribute 'area'
```

## Proposed fix

One defensive line in `compute_sa2_areas_km2`:

```python
return {
    str(code): float(geom.area / 1_000_000.0)
    for code, geom in zip(in_equal_area[code_column], in_equal_area.geometry, strict=False)
    if geom is not None   # ← add this
}
```

SA2s with null geometry get omitted from the area lookup, so any
density column that consumes the lookup falls back to NaN for those
codes — which is the right behaviour (no density for non-areal
pseudo-SA2s) and matches the existing per-SA2 null pattern.

A follow-up could also warn at construction time when a non-trivial
fraction of boundaries lack geometry, but the bare-minimum fix is the
one-line guard.

## Downstream workaround

Until this lands, downstream consumers can monkey-patch:

```python
import census_augment.spatial as _spatial_mod

def _safe_compute_sa2_areas_km2(boundaries, *, code_column="SA2_CODE21"):
    if code_column not in boundaries.columns:
        raise ValueError(f"code column {code_column!r} not found in boundaries")
    if boundaries.crs is None:
        raise ValueError("boundaries GeoDataFrame must have a CRS")
    in_equal_area = boundaries.to_crs("EPSG:3577")
    return {
        str(code): float(geom.area / 1_000_000.0)
        for code, geom in zip(
            in_equal_area[code_column], in_equal_area.geometry, strict=False
        )
        if geom is not None
    }

_spatial_mod.compute_sa2_areas_km2 = _safe_compute_sa2_areas_km2
```

Install before importing `census_augment.Pipeline`.

## Context

Encountered while building an overnight experiment runner against
`main` for the [aus-fuel-forecaster](https://github.com/cauldnz/aus-fuel-forecaster)
project — bumping the pin from `65fd3fa6` (last working) to `762a6a0f`
to pick up the fixes for cauldnz/abs-census-augmentor#91 Stage 2 and #99.
The bug is high-impact: it blocks every `Pipeline.create()` call on
current main, which is a hard-stop for any downstream project that
upgrades.
