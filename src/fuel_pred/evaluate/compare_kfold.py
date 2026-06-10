"""CLI entry point for the v3.0 k-fold comparison report.

Thin wrapper so ``python -m fuel_pred.evaluate.compare_kfold`` works.
All the logic lives in ``compare.compare_kfold`` (rendering + metrics
helpers are shared with the single-split ``compare()`` for the v2.x
path).

Usage:

    uv run python -m fuel_pred.evaluate.compare_kfold \\
        --features data/processed/features.parquet \\
        --models-root models_kfold \\
        --out results/comparison_kfold.md

Spec: spec.md §15.2 (v3.0 Phase 1).
"""
from __future__ import annotations

from fuel_pred.evaluate.compare import (
    compare_kfold as compare_kfold,
)
from fuel_pred.evaluate.compare import (
    main_kfold,
)

# Public re-export so importers can do
# ``from fuel_pred.evaluate.compare_kfold import compare_kfold``.
__all__ = ["compare_kfold"]


if __name__ == "__main__":
    main_kfold()
