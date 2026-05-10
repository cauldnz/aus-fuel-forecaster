# Pending upstream issue — file at `cauldnz/abs-census-augmentor`

This file holds the body of an issue we want filed against the augmentor.
Once filed, link it from `src/fuel_pred/build/enrich_census.py` (search for
`UPSTREAM_GCP_COLLISION`) and delete this file.

---

**Title:** `Pipeline.augment crashes when a direct GCP variable shares a code with a PRESET source`

## Summary

Requesting both a direct GCP variable (e.g. `G01.Tot_P_P`) and a PRESET that uses the same code as one of its source fields (e.g. `PRESET.pct_aged_65_plus`) crashes inside `_build_gcp_lookup` with `ValueError: cannot reindex on an axis with duplicate labels`.

## Repro (against augmentor v1.4.2)

```python
import pandas as pd
from census_augment import Pipeline

df = pd.DataFrame({'lat': [-33.85], 'lon': [151.20]})
pipe = Pipeline.create(
    variables={
        'pop':  'G01.Tot_P_P',              # direct
        'aged': 'PRESET.pct_aged_65_plus',  # uses G01.Tot_P_P as denominator
    },
    user_agent='repro/0.1',
    latitude_column='lat', longitude_column='lon',
)
pipe.augment(df, latitude_column='lat', longitude_column='lon')
```

Stack trace bottoms out in `census_augment.features.FeatureEvaluator.evaluate` at `ratio = num / den` because `den` is a 2-column DataFrame (both columns named `G01.Tot_P_P`), not a Series.

## Cause

In `census_augment.enrich.CensusEnricher._build_gcp_lookup`:

```python
codes = [code for _, code in fc]                                      # -> ['Tot_P_P', 'Tot_P_P']
rename_map = {code: f"{self._output_prefix}{friendly}" for friendly, code in fc}  # last friendly wins
pieces.append(table_df[codes].rename(columns=rename_map))             # both cols renamed to same name
```

When PRESET source-collection auto-injects `__preset_src__G01.Tot_P_P -> G01.Tot_P_P` alongside the user's `pop -> G01.Tot_P_P`, both end up in `friendly_refs` for the same GCP table. `table_df[codes]` returns a DataFrame with duplicate column labels, and `.rename()` collapses them onto whichever friendly name happened to win the dict comprehension — leaving two columns sharing one name in the workspace.

## Confirmed not to affect

- Each PRESET in isolation
- All 6 PRESETs together (no collision because PRESETs share sources via dedup)
- Direct GCP variables together (no collision because each user friendly maps to a distinct code)

It's specifically the cross of direct + PRESET when they share GCP codes. In our repo the only collision is `G01.Tot_P_P` (used by `pct_aged_65_plus`); no other v1.4.2 PRESET shares sources with the spec's direct variables.

## Suggested fix

In `_build_gcp_lookup`, dedupe codes at fetch time and project per-friendly using positional indexing or `.copy()` of the deduplicated source column:

```python
unique_codes = list(dict.fromkeys(code for _, code in fc))
table_slice = table_df[unique_codes]
pieces.append(pd.concat(
    {f"{self._output_prefix}{friendly}": table_slice[code] for friendly, code in fc},
    axis=1,
))
```

## Workaround in downstream code

Split into two `Pipeline.augment()` passes — one with the direct GCP variable, one with the PRESET — and merge results column-wise. Adds a second spatial join but works.

## Environment

- abs-census-augmentor 1.4.2
- pandas 2.x
- python 3.11
