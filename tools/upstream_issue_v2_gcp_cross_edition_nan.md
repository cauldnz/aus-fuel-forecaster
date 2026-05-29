# GCP cross-edition lookup returns NaN for 2016 release in temporal mode

## Summary

In v2.0.0 temporal-mode, GCP variables (`G\d+.*`) come back NaN for rows whose `date_column` value resolves to the **2016 GCP release**, even though `gcp_release="2016"` is set on the output row and the 2016 DataPack appears to be loaded (`releases_used` includes both `"2016"` and `"2021"`). SEIFA in the same call returns values for both releases — so the cross-edition orchestration partially works.

The likely cause: the augmentor passes the row's **Edition-3 SA2 code** (current ASGS) to the 2016 GCP DataPack, which is keyed by **Edition-2 SA2 codes** (different format). The lookup misses. The `gcp_sa2_code_source` column on the output row also reports the Edition-3 code, suggesting the cross-edition SA2 code translation isn't happening for GCP.

## Reproducer

Tested against v2.0.0 (commit `887ec011`). Single station, two dates straddling the 2016/2021 boundary:

```python
from datetime import date
import pandas as pd
from census_augment import Pipeline

df = pd.DataFrame({
    "station_id": ["s1", "s1"],
    "date": [date(2017, 6, 15), date(2023, 6, 15)],
    "lat": [-35.238316, -35.238316],   # Lyneham, ACT
    "lon": [149.140141, 149.140141],
})

pipeline = Pipeline.create(
    variables={
        "g01_total_pop": "G01.Tot_P_P",
        "g02_median_age": "G02.Median_age_persons",
        "seifa_irsd": "SEIFA.irsd_score",
    },
    user_agent="repro/1.0 (you@example.com)",
    latitude_column="lat",
    longitude_column="lon",
    date_column="date",
)
result = pipeline.augment(df)
print(result.releases_used)
print(result.df.T.to_string())
```

## Observed output

```
{'gcp': ['2016', '2021'], 'seifa': ['2016', '2021']}

                          0                 0
station_id               s1                s1
date              2017-06-15        2023-06-15
lat               -35.238316        -35.238316
lon               149.140141        149.140141
sa2_code           801051057         801051057
sa2_code_edition           3                 3
gcp_release             2016              2021
seifa_release           2016              2021
gcp_sa2_code_source 801051057         801051057   ← Edition-3 SA2 code used for 2016 release
seifa_sa2_code_source 801051057       801051057
sa2_g01_total_pop        NaN            5703.0   ← 2016 row returns NaN
sa2_g02_median_age       NaN              35.0   ← 2016 row returns NaN
sa2_seifa_irsd        1056.0       1053.481453   ← SEIFA works for both years
```

## Expected output

Either:
- The 2016 GCP DataPack lookup succeeds via Edition-2-keyed translation (correct cross-edition behaviour, matching what SEIFA appears to do), OR
- The augmentor raises a clear error explaining that GCP cross-edition lookup is not implemented for this dataset (rather than silently returning NaN with a misleading `gcp_release="2016"` annotation)

## Why this matters for downstream consumers

The temporal-mode release notes promise per-row resolution across editions. Discovering the silent NaN return mid-integration costs cycles. If GCP cross-edition translation is a known limitation, calling it out in the release notes / `temporal-mode.md` would help.

## Context

Encountered while planning a `date_column`-driven adoption of v2.0 temporal mode in a fuel-price forecasting project ([aus-fuel-forecaster](https://github.com/cauldnz/aus-fuel-forecaster), see `docs/research/2026-05_abs_census_augmentor_v2.0_review.md` for the full spike write-up). Diagnostic script lives at `tools/research/v2_spike_diagnose.py` in that repo.
