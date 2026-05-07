"""Hermetic tests for clean.fuelcheck."""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: implement once clean.fuelcheck is filled in (Phase 2)")
def test_station_id_is_stable_across_runs() -> None:
    """Same (name, address, suburb, postcode) → same station_id."""
    pass


@pytest.mark.skip(reason="TODO")
def test_brand_aliasing_canonicalises_known_variants() -> None:
    pass


@pytest.mark.skip(reason="TODO")
def test_daily_aggregation_produces_one_row_per_station_fuel_day() -> None:
    pass
