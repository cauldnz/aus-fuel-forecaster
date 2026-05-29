# ERP temporal-release resolution only sees the latest publication

## Summary

In v2.0.0, `Pipeline.augment(df, date_column="...")` with any `ERP.*` variable raises:

```
RuntimeError: ERP release '<year>' not found. Available: ['<latest_year>']
```

for rows whose `date_column` value doesn't fall in the latest publication year. The release notes and `temporal-mode.md` imply that temporal mode works for all datasets, but ABS publishes ERP differently from SEIFA / GCP / DSS: there's **one annual workbook per publication that contains the full back-series** in `population_history_<year>` columns, not separate per-year releases.

The augmentor's per-release resolver treats ERP like SEIFA (one snapshot per release year) and so fails on any row whose date doesn't match the latest publication year.

## Reproducer

Same setup as the GCP cross-edition issue, but ask for ERP:

```python
from datetime import date
import pandas as pd
from census_augment import Pipeline

df = pd.DataFrame({
    "station_id": ["s1"] * 4,
    "date": [date(2017, 6, 15), date(2020, 6, 15), date(2023, 6, 15), date(2024, 6, 15)],
    "lat": [-35.238316] * 4,
    "lon": [149.140141] * 4,
})

pipeline = Pipeline.create(
    variables={
        "erp_total": "ERP.population_total",
        "erp_65_plus": "ERP.population_65_plus",  # new in v2.0 PR #82
    },
    user_agent="repro/1.0 (you@example.com)",
    latitude_column="lat",
    longitude_column="lon",
    date_column="date",
)
result = pipeline.augment(df)
```

## Observed

```
RuntimeError: ERP release '2017' not found. Available: ['2024']
```

(With cache cold; "Available" list may differ depending on what was fetched.)

## Two ways this could be addressed

1. **Docs:** clarify in `temporal-mode.md` that ERP is cross-sectional-only (latest release) and that per-row historical population comes from `population_history_<year>` direct refs, not from the temporal resolver. Then in temporal mode, either skip ERP variables silently with a warning, or auto-fall-back to the latest release for them.

2. **Code:** model ERP as one logical release per `population_history_<year>` column (i.e. the 2024 publication exposes effective releases ['2001', '2002', ..., '2024'] with the value at each row's date pulled from the matching column). This is more work but lets ERP slot into temporal mode like other datasets.

The cross-dataset PRESETs from v2.0 PR #86 (`pct_age_pension_recipients`, `pct_jobseeker_recipients`, `welfare_density_index`) inherit this limitation because they use ERP denominators. So whichever path is taken, it should also resolve the PRESETs.

## Why this matters for downstream consumers

Discovering this mid-integration is a pivot moment — the v2.0 release notes claim cross-edition temporal augmentation works for all datasets, but ERP's structural shape is fundamentally different. A clear note in the release notes / dataset docs would save downstream consumers a debugging round.

## Context

Encountered while planning a `date_column`-driven adoption of v2.0 temporal mode in a fuel-price forecasting project ([aus-fuel-forecaster](https://github.com/cauldnz/aus-fuel-forecaster), see `docs/research/2026-05_abs_census_augmentor_v2.0_review.md` for the full spike write-up).
