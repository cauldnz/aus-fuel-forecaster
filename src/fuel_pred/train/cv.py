"""Time-series k-fold cross-validation orchestrator (spec §15.2).

Wraps the per-fold training loop. For each of ``k`` folds (split by
``train.folds.split_kfolds``), fits Models A/B/B' and persists per-fold
artefacts under ``<out_root>/fold_N/``. Aggregation lives downstream in
``evaluate.compare_kfold``; this module only writes the per-fold inputs
the comparison report consumes.

Layout:

    <out_root>/
        fold_1/
            model_a.pkl
            model_b.pkl
            model_b_prime.pkl
            feature_lists.json
            predictions_test.parquet  # the k-fold "test" slice
        fold_2/
            ...
        fold_6/
            ...
        kfold_audit.json  # k-fold config + per-fold result summary

CLI:

    uv run python -m fuel_pred.train.cv \\
        --features data/processed/features.parquet \\
        --out models_kfold

The default ``KFoldConfig`` produces the 6-fold geometry locked in
spec §15.2 (also documented at
``docs/research/2026-05_v3.0_phase1_kfold_design.md``).

Spec: spec.md §15.2 (v3.0 Phase 1).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from fuel_pred.train._fit import DEFAULT_LOG_PERIOD, FitResult
from fuel_pred.train.folds import KFoldConfig, split_kfolds
from fuel_pred.train.train_models import (
    TARGET_COLUMN,
    _load_and_filter_target,
    _train_one_fold,
)

logger = logging.getLogger(__name__)


@dataclass
class KFoldRunResult:
    """Aggregate result of a ``train_kfold`` invocation.

    - ``per_fold_results``: list of ``{model_id: FitResult}`` dicts in
      fold order (1..k).
    - ``per_fold_out_dirs``: paths to each fold's persistence dir.
    - ``kfold_config``: the geometry the run used.
    """

    per_fold_results: list[dict[str, FitResult]]
    per_fold_out_dirs: list[Path]
    kfold_config: KFoldConfig


def train_kfold(
    features_path: Path,
    out_root: Path,
    *,
    kfold_config: KFoldConfig | None = None,
    target: str = TARGET_COLUMN,
    save_predictions: bool = True,
    log_period: int = DEFAULT_LOG_PERIOD,
    n_estimators: int | None = None,
    random_state: int | None = None,
    models_to_fit: tuple[str, ...] = ("A", "B", "B_PRIME"),
) -> KFoldRunResult:
    """Run k-fold CV: fit A/B/B' once per fold; write per-fold artefacts.

    Args:
        features_path: ``data/processed/features.parquet``.
        out_root: parent dir for per-fold subdirs. Created if missing.
        kfold_config: geometry; defaults to ``KFoldConfig.default()``
            (the spec §15.2 6-fold scheme).
        target / save_predictions / log_period / n_estimators / random_state /
            models_to_fit: forwarded to ``_train_one_fold`` — same semantics
            as ``train()``. ``random_state`` and ``models_to_fit`` are used
            by the v3.0 Phase 3 seed-noise experiment (next-step #2): pass
            a per-run seed + ``("A",)`` to fit only Model A per seed-run,
            cutting wall-clock by ~2/3.

    Returns:
        ``KFoldRunResult`` capturing per-fold ``FitResult`` dicts +
        output paths + the config that drove the split.

    Side effects:
        - Per-fold pickles, ``feature_lists.json``, and (optionally)
          ``predictions_test.parquet`` under ``out_root/fold_N/``.
        - ``out_root/kfold_audit.json`` summary at the end.
    """
    cfg = kfold_config or KFoldConfig.default()
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "k-fold CV starting: k=%d, test_window_months=%d, val_window_days=%d, "
        "gap_days=%d, horizon_days=%d, panel_end=%s, warmup_end=%s",
        cfg.k, cfg.test_window_months, cfg.val_window_days,
        cfg.gap_days, cfg.horizon_days, cfg.panel_end, cfg.warmup_end,
    )

    work = _load_and_filter_target(features_path, target)
    folds = split_kfolds(work, kfold_config=cfg)

    per_fold_results: list[dict[str, FitResult]] = []
    per_fold_out_dirs: list[Path] = []
    for fold_idx, fold in enumerate(folds, start=1):
        fold_out = out_root / f"fold_{fold_idx}"
        logger.info("=" * 70)
        logger.info(
            "kfold fold %d/%d: train=%d val=%d test=%d (out=%s)",
            fold_idx, cfg.k,
            len(fold["train"]), len(fold["val"]), len(fold["test"]),
            fold_out,
        )
        logger.info("=" * 70)

        # _train_one_fold takes a {test_fold_name: df} dict so it can
        # write per-fold prediction parquets. For k-fold, the only test
        # slice is the fold's own "test" window.
        result = _train_one_fold(
            train_full=fold["train"],
            val_full=fold["val"],
            test_folds={"test": fold["test"]},
            out_dir=fold_out,
            target=target,
            save_predictions=save_predictions,
            log_period=log_period,
            n_estimators=n_estimators,
            random_state=random_state,
            models_to_fit=models_to_fit,
        )
        per_fold_results.append(result)
        per_fold_out_dirs.append(fold_out)

    _write_kfold_audit(
        out_root / "kfold_audit.json",
        cfg,
        per_fold_results,
        per_fold_out_dirs,
    )
    logger.info("k-fold CV complete: %d folds → %s", cfg.k, out_root)
    return KFoldRunResult(
        per_fold_results=per_fold_results,
        per_fold_out_dirs=per_fold_out_dirs,
        kfold_config=cfg,
    )


def _write_kfold_audit(
    path: Path,
    cfg: KFoldConfig,
    per_fold_results: list[dict[str, FitResult]],
    per_fold_out_dirs: list[Path],
) -> None:
    """Persist a top-level summary of the run.

    Lets ``evaluate.compare_kfold`` enumerate fold dirs + best-iteration
    audit without re-loading every pickle. Schema mirrors per-fold
    feature_lists.json plus the KFoldConfig that drove the geometry.
    """
    fold_summaries = []
    for fold_idx, (result, fold_dir) in enumerate(
        zip(per_fold_results, per_fold_out_dirs, strict=True), start=1
    ):
        fold_summaries.append({
            "fold": fold_idx,
            "out_dir": str(fold_dir),
            "models": {
                model_id: {
                    "best_iteration": fit.best_iteration,
                    "best_val_mae": fit.best_score,
                    "n_features": len(fit.feature_columns),
                    "n_categorical": len(fit.categorical_columns),
                }
                for model_id, fit in result.items()
            },
        })
    payload = {
        "kfold_config": {
            "k": cfg.k,
            "test_window_months": cfg.test_window_months,
            "val_window_days": cfg.val_window_days,
            "gap_days": cfg.gap_days,
            "horizon_days": cfg.horizon_days,
            "warmup_end": cfg.warmup_end,
            "panel_end": cfg.panel_end,
        },
        "folds": fold_summaries,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)


# ---- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help="target column (default y_t1; y_t1_t7 also valid)",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="skip writing per-fold prediction parquets",
    )
    parser.add_argument(
        "--log-period",
        type=int,
        default=DEFAULT_LOG_PERIOD,
        help="LightGBM eval-line cadence (default %(default)s; 0 = silent)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="cap on boosting rounds per model per fold (default: spec §8.2)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="override KFoldConfig.k (default: 6 per spec §15.2)",
    )
    parser.add_argument(
        "--test-window-months",
        type=int,
        default=None,
        help="override test window width in months (default: 12)",
    )
    parser.add_argument(
        "--gap-days",
        type=int,
        default=None,
        help=(
            "override gap_days between train and test (default: 1 to "
            "prevent the y_t1 target-shift leak)"
        ),
    )
    parser.add_argument(
        "--panel-end",
        default=None,
        help="override panel end date (YYYY-MM-DD; default 2026-04-30)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    # Build a KFoldConfig from overrides, falling back to defaults
    overrides: dict = {}
    if args.k is not None:
        overrides["k"] = args.k
    if args.test_window_months is not None:
        overrides["test_window_months"] = args.test_window_months
    if args.gap_days is not None:
        overrides["gap_days"] = args.gap_days
    if args.panel_end is not None:
        overrides["panel_end"] = args.panel_end
    cfg = KFoldConfig(**overrides) if overrides else KFoldConfig.default()

    train_kfold(
        args.features,
        args.out,
        kfold_config=cfg,
        target=args.target,
        save_predictions=not args.no_predictions,
        log_period=args.log_period,
        n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
