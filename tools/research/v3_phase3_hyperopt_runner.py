"""v3.0 Phase 3 next-step #4 — Optuna hyperparameter sweep for Model A.

Reading C1 test from the Phase 2 postmortem: is the spec §8.2 hyperparameter
config (num_leaves=63, min_data_in_leaf=200, learning_rate=0.05, ...) the
right capacity for Model A, or are we leaving meaningful MAE on the table?

The Phase 3 #1/#2/#3 experiments established that the v2.x augmentor surface
is dead-on-arrival for this model class — Model A is what we ship. This
experiment asks: can we improve Model A itself with a proper hyperparameter
search? If yes, the ship-Model-A decision lands with a stronger model than
the spec default.

Method:

- **Optuna TPE sampler** (Bayesian) over the LightGBM hyperparameter space.
- **Per-trial: 6-fold k-fold Model A only**, mean val-MAE as objective.
- **MedianPruner**: after each fold report, prune trials performing below
  median of the same-step running mean of completed trials.
- **Resume-safe**: SQLite-backed study survives kill/restart. Same study
  name on restart picks up where killed.
- **Wall-clock budget**: 8 hours (overnight). With pruning, expect
  120-180 trials (TPE converges by ~50-80 typically).

Search space (all centred on or expanding around spec §8.2 defaults):

- num_leaves: {15, 31, 63, 127, 255} — capacity / depth proxy
- min_data_in_leaf: 50-1000 log-uniform — regularization
- learning_rate: 0.005-0.15 log-uniform
- feature_fraction: 0.4-1.0 uniform — bagging columns
- bagging_fraction: 0.4-1.0 uniform — bagging rows
- bagging_freq: {0, 1, 5, 10} categorical
- lambda_l1: 1e-8 to 10.0 log-uniform — sparsity reg
- lambda_l2: 1e-8 to 10.0 log-uniform — weight decay reg

Fixed (spec invariants):
- objective=regression_l1, metric=mae, random_state=42
- n_estimators=2000 with early_stopping_rounds=100 — early stopping
  controls effective tree count per trial automatically

Run:

    uv run python tools/research/v3_phase3_hyperopt_runner.py \
        --hours 8 \
        --n-trials 200 \
        2>&1 | tee tools/research/v3_phase3_hyperopt.tee.log

Spec / discussion: ``docs/research/2026-06_v3.0_phase2_postmortem_discussion.md``
(next-step #4 in the ranked list — Reading C1 capacity test).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Optuna's own logging is noisy at INFO; route through our handler at WARNING
# so trial summary lines are visible but per-trial-suggestion noise isn't.
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

# SQLite study + per-trial scratch dir (recycled each trial — no disk
# footprint that scales with N_TRIALS).
STUDY_DIR = REPO_ROOT / "models_optuna_a"
STUDY_DIR.mkdir(parents=True, exist_ok=True)
STUDY_DB = STUDY_DIR / "study.db"
SCRATCH_DIR = STUDY_DIR / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
STUDY_NAME = "phase3_model_a_hyperopt"

FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"

LOG_PATH = REPO_ROOT / "tools" / "research" / "v3_phase3_hyperopt.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a"),
    ],
)
logger = logging.getLogger("v3_phase3_hyperopt")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _import_train_modules() -> tuple[Any, ...]:
    """Late import so logging is set up first."""
    from fuel_pred import config
    from fuel_pred.train.folds import KFoldConfig, split_kfolds
    from fuel_pred.train.train_models import _load_and_filter_target, _train_one_fold
    return config, KFoldConfig, split_kfolds, _load_and_filter_target, _train_one_fold


def _suggest_params(trial: optuna.Trial) -> dict[str, object]:
    """Sample one point in the LightGBM hyperparameter space."""
    return {
        # Capacity / depth proxy. Categorical so TPE handles modes well
        # rather than treating it as a continuous integer.
        "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63, 127, 255]),
        # Regularization — larger = less overfitting on rare-feature splits.
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000, log=True),
        # Learning rate (log scale spans 30x — small to aggressive).
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        # Column subsampling per tree.
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        # Row subsampling.
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        # Frequency of bagging. 0 = disabled (bagging_fraction unused).
        "bagging_freq": trial.suggest_categorical("bagging_freq", [0, 1, 5, 10]),
        # L1 / L2 regularization on leaf weights.
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def _objective_factory(
    modules: tuple[Any, ...], folds: list[dict[str, Any]]
) -> Callable[[optuna.Trial], float]:
    """Return a closure that evaluates one trial across all 6 folds.

    The closure captures the in-memory ``folds`` (built once at study start)
    so each trial avoids re-loading the 15M-row features parquet (~25s I/O
    saved per trial). ``modules`` is the late-imported tuple of fuel_pred
    bits; same idea — pay the import cost once.
    """
    config, _kfc, _split, _load, _train_one_fold = modules

    def objective(trial: optuna.Trial) -> float:
        suggested = _suggest_params(trial)
        # Monkey-patch config.LGBM_PARAMS so _train_one_fold's
        # ``fit_params = dict(config.LGBM_PARAMS)`` pickup the new values.
        # Snapshot + restore in a finally clause so a crash doesn't leave
        # config dirty for the next trial.
        snapshot = dict(config.LGBM_PARAMS)
        config.LGBM_PARAMS.update(suggested)
        # Log the trial start with a compact param summary
        compact = (
            f"nl={suggested['num_leaves']:>3d} "
            f"min_data={suggested['min_data_in_leaf']:>4d} "
            f"lr={suggested['learning_rate']:.4f} "
            f"ff={suggested['feature_fraction']:.2f} "
            f"bf={suggested['bagging_fraction']:.2f}/{suggested['bagging_freq']} "
            f"l1={suggested['lambda_l1']:.2e} l2={suggested['lambda_l2']:.2e}"
        )
        t_trial = time.monotonic()
        logger.info("trial %3d start | %s", trial.number, compact)
        try:
            fold_maes: list[float] = []
            for fold_idx, fold in enumerate(folds, start=1):
                t_fold = time.monotonic()
                result = _train_one_fold(
                    train_full=fold["train"],
                    val_full=fold["val"],
                    test_folds={},  # no test preds — only val MAE matters
                    out_dir=SCRATCH_DIR,  # recycled per fold; ~20MB on disk
                    save_predictions=False,
                    models_to_fit=("A",),
                )
                fold_mae = result["A"].best_score
                if fold_mae is None:
                    raise RuntimeError(f"trial {trial.number} fold {fold_idx}: best_score is None")
                fold_maes.append(float(fold_mae))
                running_mean = statistics.fmean(fold_maes)
                logger.info(
                    "trial %3d fold %d/%d | val_mae=%.4f running_mean=%.4f (%.0fs)",
                    trial.number, fold_idx, len(folds),
                    fold_mae, running_mean, time.monotonic() - t_fold,
                )
                # Report to optuna for pruning — step is fold index, value
                # is the running mean MAE across folds completed so far.
                trial.report(running_mean, fold_idx)
                if trial.should_prune():
                    elapsed = time.monotonic() - t_trial
                    logger.info(
                        "trial %3d PRUNED at fold %d/%d (mean=%.4f, %.0fs)",
                        trial.number, fold_idx, len(folds),
                        running_mean, elapsed,
                    )
                    raise optuna.TrialPruned()

            final_mean = statistics.fmean(fold_maes)
            elapsed = time.monotonic() - t_trial
            logger.info(
                "trial %3d DONE  | mean_val_mae=%.4f (per-fold %s) [%.0fs]",
                trial.number, final_mean,
                [f"{m:.3f}" for m in fold_maes], elapsed,
            )
            return final_mean
        finally:
            config.LGBM_PARAMS.clear()
            config.LGBM_PARAMS.update(snapshot)

    return objective


def _write_summary(study: optuna.Study, ref_params: dict) -> None:
    """Write the hyperopt summary doc + JSON dump of all trials."""
    summary_md = RESULTS_DIR / "v3_phase3_hyperopt_summary.md"
    summary_json = RESULTS_DIR / "v3_phase3_hyperopt.json"

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    payload = {
        "study_name": study.study_name,
        "n_trials_total": len(study.trials),
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "n_failed": len(failed),
        "best_value": study.best_value if completed else None,
        "best_params": study.best_params if completed else None,
        "reference_default_params": ref_params,
        "completed_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "duration_s": t.duration.total_seconds() if t.duration else None,
            }
            for t in sorted(completed, key=lambda x: x.value or float("inf"))
        ],
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_json)

    if not completed:
        summary_md.write_text(
            "# v3.0 Phase 3 #4 — Model A hyperparameter sweep\n\n"
            f"Study `{study.study_name}` had no completed trials yet "
            f"({len(study.trials)} attempted, {len(pruned)} pruned, "
            f"{len(failed)} failed).\n", encoding="utf-8",
        )
        return

    best_val = study.best_value
    ref_default_val = None
    # Reference default = the spec §8.2 defaults; mean val MAE across the
    # 6 folds at those defaults was measured in the v3.0 Phase 3 seed-noise
    # run for seed=42 (the spec default seed). Pull that to anchor the
    # "did we improve over the spec default?" question.
    seed_noise_path = RESULTS_DIR / "v3_phase3_seed_noise.json"
    if seed_noise_path.exists():
        seed_noise = json.loads(seed_noise_path.read_text(encoding="utf-8"))
        seed42 = seed_noise["per_seed_per_fold_mae_a"].get("42", {})
        fold_vals = [v for k, v in seed42.items() if k.startswith("fold_")]
        if fold_vals:
            ref_default_val = statistics.fmean(fold_vals)

    lines: list[str] = []
    lines.append("# v3.0 Phase 3 #4 — Model A hyperparameter sweep (Reading C1)")
    lines.append("")
    lines.append(
        "Optuna TPE Bayesian search over Model A's LightGBM hyperparameters, "
        "evaluated on the v3.0 6-fold k-fold harness (mean val-MAE). "
        f"Study: `{study.study_name}`."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Best mean val-MAE: {best_val:.4f} c/L**")
    if ref_default_val is not None:
        delta = best_val - ref_default_val
        improvement_pct = 100 * (ref_default_val - best_val) / ref_default_val
        lines.append(
            f"- Reference (spec §8.2 defaults, seed 42): **{ref_default_val:.4f} c/L**"
        )
        lines.append(
            f"- **Improvement: {-delta:+.4f} c/L ({improvement_pct:+.2f}%)** — "
            f"{'WIN' if delta < 0 else 'no improvement'} over spec default"
        )
    lines.append(f"- Trials completed: {len(completed)}")
    lines.append(f"- Trials pruned: {len(pruned)}")
    if failed:
        lines.append(f"- Trials failed: {len(failed)}")
    lines.append("")

    lines.append("## Best hyperparameters")
    lines.append("")
    lines.append("| Hyperparameter | Best | Spec §8.2 default |")
    lines.append("|----------------|------|-------------------|")
    spec_defaults = {
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
    }
    best_params = study.best_params
    for k in (
        "num_leaves", "min_data_in_leaf", "learning_rate",
        "feature_fraction", "bagging_fraction", "bagging_freq",
        "lambda_l1", "lambda_l2",
    ):
        if k not in best_params:
            continue
        v = best_params[k]
        d = spec_defaults.get(k, "—")
        v_s = f"{v:.4g}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {v_s} | {d} |")
    lines.append("")

    # Top-10 trials by value
    top = sorted(completed, key=lambda x: x.value)[:10]
    lines.append("## Top 10 trials")
    lines.append("")
    lines.append("| Rank | Trial | Mean val-MAE | num_leaves | min_data | lr | ff | bf/bf_freq | l1 | l2 |")
    lines.append("|-----:|------:|-------------:|-----------:|---------:|---:|---:|-----------:|---:|---:|")
    for rank, t in enumerate(top, start=1):
        p = t.params
        lines.append(
            f"| {rank} | {t.number} | {t.value:.4f} | "
            f"{p.get('num_leaves', '?')} | {p.get('min_data_in_leaf', '?')} | "
            f"{p.get('learning_rate', 0):.4f} | "
            f"{p.get('feature_fraction', 0):.2f} | "
            f"{p.get('bagging_fraction', 0):.2f}/{p.get('bagging_freq', '?')} | "
            f"{p.get('lambda_l1', 0):.2e} | {p.get('lambda_l2', 0):.2e} |"
        )
    lines.append("")

    lines.append("## Reading")
    lines.append("")
    if ref_default_val is not None:
        if best_val < ref_default_val - 0.05:
            lines.append(
                f"**Spec default beaten by {-delta:.3f} c/L "
                f"({improvement_pct:+.1f}%)** — meaningful capacity to unlock. "
                "Recommend updating spec §8.2 with the new defaults before "
                "shipping Model A. Validate the chosen params with a 6× seed "
                "sanity check (re-run Phase 3 #2's seed-noise floor with "
                "the new params) before locking them in."
            )
        elif best_val < ref_default_val - 0.01:
            lines.append(
                f"**Marginal improvement: {-delta:.3f} c/L** — within the "
                "seed-noise floor (~0.09 c/L per-fold stdev from Phase 3 #2). "
                "Not a clear win; the spec §8.2 defaults are already near-"
                "optimal for this feature set. Keep spec defaults; ship as-is."
            )
        else:
            lines.append(
                f"**No improvement vs spec default** (best {best_val:.4f} vs "
                f"{ref_default_val:.4f}). LightGBM is at capacity for this "
                "feature set — Reading C1 (hyperparameter mismatch) falsified. "
                "Confirms the Phase 3 conclusion: ship Model A on spec §8.2 "
                "defaults; the augmentor surface, model class, and "
                "hyperparameter space have all been tested and none unlock "
                "meaningful additional signal."
            )
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append("- `tools/research/v3_phase3_hyperopt_runner.py` — this script")
    lines.append(f"- `{STUDY_DB.relative_to(REPO_ROOT)}` — Optuna SQLite study (gitignored)")
    lines.append("- `docs/research/2026-06_v3.0_phase2_postmortem_discussion.md` — Reading C1 hypothesis")
    lines.append("- `results/v3_phase3_seed_noise_summary.md` — reference for spec-default MAE")
    lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", summary_md)
    logger.info("=== Hyperopt headline ===")
    logger.info("Best mean val-MAE: %.4f c/L", best_val)
    if ref_default_val is not None:
        logger.info("Spec default:      %.4f c/L", ref_default_val)
        logger.info("Improvement:       %+.4f c/L (%+.2f%%)", -delta, improvement_pct)
    logger.info("Trials completed:  %d (+ %d pruned)", len(completed), len(pruned))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials", type=int, default=200,
        help="max trials (study stops at min of n_trials and hours budget)",
    )
    parser.add_argument(
        "--hours", type=float, default=8.0,
        help="wall-clock budget in hours (default 8 = overnight)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="delete the SQLite study + scratch dir and start fresh",
    )
    args = parser.parse_args()

    if args.reset and STUDY_DB.exists():
        logger.info("--reset: removing %s", STUDY_DB)
        STUDY_DB.unlink()
    if args.reset and SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR)
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("v3.0 Phase 3 hyperopt starting (n_trials<=%d, %.1fh budget)",
                args.n_trials, args.hours)

    modules = _import_train_modules()
    config, kfold_config_cls, split_kfolds, load_and_filter, _train_one_fold = modules
    ref_params = dict(config.LGBM_PARAMS)
    logger.info("spec §8.2 default params: %s", ref_params)

    logger.info("loading features + splitting folds (one-time)")
    t0 = time.monotonic()
    work = load_and_filter(FEATURES_PARQUET, "y_t1")
    folds = split_kfolds(work, kfold_config=kfold_config_cls.default())
    logger.info("loaded + split %d folds in %.1fs", len(folds), time.monotonic() - t0)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB.as_posix()}",
        sampler=TPESampler(seed=42, n_startup_trials=10),
        # Don't prune the first 5 trials (need a reference distribution),
        # and don't prune before fold 2 (1 fold isn't a stable signal).
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        direction="minimize",
        load_if_exists=True,
    )
    logger.info(
        "study %r ready: %d existing trials (best so far: %s)",
        STUDY_NAME, len(study.trials),
        f"{study.best_value:.4f}" if study.trials and any(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        ) else "n/a",
    )

    objective = _objective_factory(modules, folds)
    timeout = int(args.hours * 3600)

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=timeout,
            gc_after_trial=True,
            callbacks=[
                # Re-write the summary after every completed/pruned trial
                # so a kill mid-run preserves the latest snapshot.
                lambda _s, _t: _write_summary(study, ref_params) if (_t.number + 1) % 5 == 0 else None,
            ],
        )
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — writing final summary then exiting")

    _write_summary(study, ref_params)
    logger.info("v3.0 Phase 3 hyperopt complete")


if __name__ == "__main__":
    main()
