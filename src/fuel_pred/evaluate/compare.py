"""Produce results/comparison.md — the headline result document.

Compares Models A and B across the test folds, with segmentation by:
    - metro / regional
    - top 8 brands + Other
    - fuel type (U91, DL)
    - SEIFA quintile

Reports MAE, RMSE, MAPE, median absolute error, p90 absolute error.

Outputs results/comparison.md in a clean Markdown table format that renders
nicely on GitHub.

Spec: spec.md §8.5, §12 Phase 6.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compare(features_path: Path, models_dir: Path, out_path: Path) -> None:
    raise NotImplementedError("TODO: implement per spec.md §8.5.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    compare(args.features, args.models, args.out)


if __name__ == "__main__":
    main()
