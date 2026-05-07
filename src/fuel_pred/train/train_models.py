"""Fit Models A (no SA2) and B (with SA2) on the feature matrix.

Both models use identical hyperparameters (config.LGBM_PARAMS) and identical
training rows: only rows where every Model B column is non-null are used in
EITHER model. This prevents the augmentor from looking better just because
its richer column set excluded harder examples.

Splits per spec.md §8.3:
    Train: <= TRAIN_END
    Validation (early stopping): VAL_START..VAL_END
    Test (normal): TEST_START..TEST_NORMAL_END
    Test (crisis): >= TEST_CRISIS_START   (held out, reported separately)

Outputs:
    models/model_a.pkl
    models/model_b.pkl
    models/feature_lists.json   (the actual columns each model used)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def train(features_path: Path, models_dir: Path) -> None:
    raise NotImplementedError("TODO: implement per spec.md §8.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train(args.features, args.out)


if __name__ == "__main__":
    main()
