"""Pipeline modules that build feature inputs from cleaned interim data.

Per spec.md §10:
- panel_grid.py    — assemble the (station, fuel, date) grid (Phase 4)
- enrich_census.py — call abs-census-augmentor → SA2 + sa2_* features (Phase 3)
- make_features.py — implement all feature blocks from §7 (Phase 4)
"""
from __future__ import annotations
