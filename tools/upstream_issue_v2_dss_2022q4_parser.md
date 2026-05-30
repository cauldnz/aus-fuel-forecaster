# DSS XLSX parser fails on 2022-Q4 release: "No SA2 data rows"

## Summary

In v2.0+ temporal mode, asking for any `DSS.*` variable with a row whose
date predates 2022-Q4 (using `temporal.out_of_range: nearest`) causes the
augmentor to fetch and parse `dss-2022-Q4.xlsx` as the earliest
available release. The XLSX downloads cleanly, but the parser raises:

```
RuntimeError: No SA2 data rows in <cache>/dss_payments/dss-2022-Q4.xlsx
```

Source: `census_augment/datasets/_dss.py:228` (`_parse_xlsx`).

The 2022-Q4 file uses a different sheet/header layout than later
quarterly releases (which our project consumed cleanly via
cross-sectional mode pointing at 2026-Q1). The parser doesn't yet handle
the older layout.

## Reproducer

Against `main` at `65fd3fa6`:

```python
from datetime import date
import pandas as pd
from census_augment import Pipeline
from census_augment.config import (
    Config, DataSourcesConfig, GeocodingConfig, InputConfig,
    NominatimConfig, OutputConfig, TemporalConfig,
)

df = pd.DataFrame({
    "station_id": ["s1"],
    "date": [date(2020, 6, 15)],   # well before 2022-Q4
    "lat": [-33.8568],
    "lon": [151.2153],
})

cfg = Config(
    input=InputConfig(latitude_column="lat", longitude_column="lon", date_column="date"),
    output=OutputConfig(prefix="sa2_"),
    geocoding=GeocodingConfig(
        providers=["nominatim"],
        nominatim=NominatimConfig(user_agent="repro/1.0 (you@example.com)"),
    ),
    data_sources=DataSourcesConfig(),
    variables={"age_pension": "DSS.age_pension_recipients"},
    temporal=TemporalConfig(out_of_range="nearest"),
)
pipeline = Pipeline.from_config(cfg)
result = pipeline.augment(df)  # raises RuntimeError
```

## Observed

```
RuntimeError: No SA2 data rows in <cache>/dss_payments/dss-2022-Q4.xlsx
```

## Impact for downstream consumers

Blocks adoption of temporal mode for DSS — the most quarterly-cadence-friendly
dataset in the augmentor surface — for any panel that spans pre-2022-Q4 dates,
which is most fuel-pricing / consumer-spending / welfare-econometrics use cases.

For our [aus-fuel-forecaster](https://github.com/cauldnz/aus-fuel-forecaster)
project (PR B / spec §7.7.2): we wanted SEIFA + DSS + ERP-total in the temporal
pass; this issue forced us to keep DSS cross-sectional for now. Code workaround
is a one-line move once the parser handles 2022-Q4. SEIFA + ERP-total temporal
work fine.

## Possible fixes

1. **Quarter-aware parsing branch.** Detect the sheet/header layout per release
   and dispatch to the right reader. Simplest if the older layout is one of a
   small handful of historical variants.
2. **Skip un-parseable releases with a WARNING.** If the augmentor can't parse a
   given release, drop it from the available set rather than raising — let
   `temporal.out_of_range: nearest` clamp to the next-oldest parseable release.
3. **Document which quarters are parseable.** If older releases are out of scope
   long-term, narrow the registered release list to the parseable subset.

## Context

Encountered after the GCP / ERP fixes (#91 Stage 1 / #92) cleared, while
building the temporal-mode pass in aus-fuel-forecaster. Spike notes in
[`docs/research/2026-05_abs_census_augmentor_v2.0_review.md`](https://github.com/cauldnz/aus-fuel-forecaster/blob/main/docs/research/2026-05_abs_census_augmentor_v2.0_review.md)
will be updated to reference this issue once filed.
