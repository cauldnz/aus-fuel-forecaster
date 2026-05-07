"""Hermetic tests for fetch.fuelcheck.

Use `responses` to mock the data.nsw.gov.au CKAN endpoints and the monthly
CSV downloads.
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: implement once fetch.fuelcheck is filled in (Phase 1)")
def test_fetches_only_months_in_range() -> None:
    pass


@pytest.mark.skip(reason="TODO")
def test_skips_cached_files_unless_force() -> None:
    pass


@pytest.mark.skip(reason="TODO")
def test_handles_schema_drift_gracefully() -> None:
    pass
