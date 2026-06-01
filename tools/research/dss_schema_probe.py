"""Probe each DSS quarterly release to map per-column coverage.

E1 (DSS temporal) failed because the augmentor's temporal-mode validation
requires every requested column to exist in every release. The error
mentioned `family_tax_benefit_a/b_recipients` were missing somewhere.
This script constructs ``DssDataSource(release='YYYY-Qn')`` for each
quarter the augmentor knows about and prints column coverage so we can
pick the universal subset and drop the rest from the temporal pass.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from census_augment.datasets._dss import DssDataSource

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("dss_probe")
logger.setLevel(logging.INFO)

DSS_CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"

# v2.1.0 release notes (#99 / PR #100) say coverage now goes 2014-Q3
# through current. Probe a sample of quarters spanning that range —
# enough to detect the FTB-A/B onset year.
QUARTERS_TO_PROBE = [
    "2014-Q3", "2015-Q1", "2016-Q1", "2017-Q1", "2018-Q1", "2019-Q1",
    "2020-Q1", "2021-Q1", "2022-Q1", "2022-Q4",
    "2023-Q1", "2023-Q2", "2023-Q4", "2024-Q2", "2024-Q4",
    "2025-Q1", "2025-Q4", "2026-Q1",
]

OUR_DSS_COLUMNS = (
    "age_pension_recipients",
    "jobseeker_payment_recipients",
    "disability_support_pension_recipients",
    "parenting_payment_single_recipients",
    "parenting_payment_partnered_recipients",
    "carer_payment_recipients",
    "carer_allowance_recipients",
    "youth_allowance_other_recipients",
    "youth_allowance_student_and_apprentice_recipients",
    "commonwealth_rent_assistance_recipients",
    "commonwealth_seniors_health_card_recipients",
    "family_tax_benefit_a_recipients",
    "family_tax_benefit_b_recipients",
)


def main() -> None:
    coverage: dict[str, set[str]] = {}  # release -> column set (only those we asked about)
    full_cols: dict[str, list[str]] = {}  # release -> sorted full col list (debug aid)
    for release in QUARTERS_TO_PROBE:
        try:
            ds = DssDataSource(release=release, root=DSS_CACHE_ROOT)
            df = ds.load()
            cols = set(df.columns)
            full_cols[release] = sorted(cols)
            coverage[release] = cols & set(OUR_DSS_COLUMNS)
            print(
                f"release={release}: {len(cols):>3} cols total | "
                f"{len(coverage[release])}/{len(OUR_DSS_COLUMNS)} of OUR cols present"
            )
        except Exception as exc:
            print(f"release={release}: FAILED - {type(exc).__name__}: {exc}")

    print()
    print(f"=== Per-column presence across {len(coverage)} probed releases ===")
    universal: list[str] = []
    partial: dict[str, list[str]] = {}
    for col in OUR_DSS_COLUMNS:
        present_in = [r for r, c in coverage.items() if col in c]
        missing_in = [r for r, c in coverage.items() if col not in c]
        marker = "ALL " if not missing_in else f"MISS"
        print(f"  [{marker}] {col:<55} present in {len(present_in)}/{len(coverage)}")
        if missing_in:
            partial[col] = missing_in
        else:
            universal.append(col)

    print()
    print(f"=== Universally available ({len(universal)}/{len(OUR_DSS_COLUMNS)}) ===")
    for col in universal:
        print(f"  + {col}")
    if partial:
        print()
        print(f"=== Partial / not in all releases ({len(partial)}/{len(OUR_DSS_COLUMNS)}) ===")
        for col, missing in partial.items():
            print(f"  - {col}")
            print(f"      missing in: {missing}")

    # Also show our 2026-Q1 union vs older release union — what's NEW since when?
    if "2026-Q1" in full_cols and "2014-Q3" in full_cols:
        new_2026 = set(full_cols["2026-Q1"]) - set(full_cols["2014-Q3"])
        print()
        print(f"=== Columns present in 2026-Q1 but not in 2014-Q3 (i.e. added later) ===")
        for col in sorted(new_2026):
            print(f"  ! {col}")


if __name__ == "__main__":
    main()
