"""Fit Models A (no SA2) and B (with SA2) on the feature matrix.

Both models use identical hyperparameters (``config.LGBM_PARAMS``) and
identical training rows: only rows where every Model B column is non-null
are used in EITHER model. This prevents the augmentor from looking better
just because its richer column set excluded harder examples.

Splits per spec.md §8.3 (delegated to ``train.folds.split_folds``).

Outputs (under ``out_dir``, typically ``models/``):
    model_a.pkl                          # pickled LGBMRegressor
    model_b.pkl                          # pickled LGBMRegressor
    feature_lists.json                   # column lists per model + audit
    predictions_test_normal.parquet      # both models' preds on the headline test fold
    predictions_test_crisis.parquet      # both models' preds on the crisis fold
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from fuel_pred import config
from fuel_pred.train._fit import FitResult, fit_lgbm
from fuel_pred.train.feature_blocks import (
    MODEL_A_BLOCKS,
    MODEL_B_BLOCKS,
    categorical_columns,
    feature_columns,
)
from fuel_pred.train.folds import FoldConfig, split_folds

logger = logging.getLogger(__name__)

TARGET_COLUMN: str = "y_t1"


def train(
    features_path: Path,
    out_dir: Path,
    *,
    fold: FoldConfig | None = None,
    target: str = TARGET_COLUMN,
    save_predictions: bool = True,
) -> dict[str, FitResult]:
    """Fit Models A and B; persist artefacts under ``out_dir``.

    Args:
        features_path: ``data/processed/features.parquet`` from
            ``build.make_features``.
        out_dir: typically ``models/``. Created if missing.
        fold: optional override of the spec §8.3 fold boundaries
            (tests pass a synthetic FoldConfig).
        target: target column name; default ``y_t1`` per spec §7.8.
            ``y_t1_t7`` is also valid for the longer-horizon variant.
        save_predictions: if True (default), also writes per-fold
            prediction parquets so the comparison-report writer
            (Phase 8) doesn't need to re-load the models.

    Returns:
        ``{"A": FitResult, "B": FitResult}`` for downstream callers
        that want the in-memory models.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load + filter to U91 rows with non-null target -----------------
    features = pd.read_parquet(features_path)
    logger.info(
        "loaded features: %d rows x %d cols", len(features), len(features.columns)
    )

    work = features[(features["fuel_code"] == "U91") & features[target].notna()].copy()
    logger.info(
        "U91 + non-null %s: %d rows (%.1f%% of input)",
        target,
        len(work),
        100 * len(work) / max(len(features), 1),
    )
    if work.empty:
        raise RuntimeError(
            f"no rows after U91+target filter; check that {features_path} has "
            f"the target column {target!r} populated"
        )

    # ---- Split into the four time-based folds ----------------------------
    folds = split_folds(work, fold=fold)

    # ---- Pick feature columns per model variant --------------------------
    # Lax mode: warn if the spec promises a column that build/make_features
    # doesn't actually emit, but proceed with whatever's there. Strict mode
    # is intended for callers (notebooks / interactive use) that want the
    # spec drift to surface as a hard error; the production training
    # pipeline should be defensive about known-pending feature columns.
    _warn_on_missing_blocks(work, MODEL_B_BLOCKS)
    cols_a = feature_columns(work, MODEL_A_BLOCKS, strict=False)
    cols_b = feature_columns(work, MODEL_B_BLOCKS, strict=False)
    cat_a = categorical_columns(cols_a)
    cat_b = categorical_columns(cols_b)
    logger.info(
        "feature counts: Model A = %d (%d categorical) ; Model B = %d (%d categorical)",
        len(cols_a),
        len(cat_a),
        len(cols_b),
        len(cat_b),
    )

    # ---- Identical-rows guard (spec §8.4) --------------------------------
    # Both models train on rows where every Model B column is non-null.
    # This makes the comparison apples-to-apples — Model B doesn't get an
    # easier dataset just because the SA2 join filtered out hard rows.
    train_full = folds["train"]
    val_full = folds["val"]
    train_mask = train_full[cols_b].notna().all(axis=1)
    val_mask = val_full[cols_b].notna().all(axis=1)
    train_eligible = train_full.loc[train_mask].copy()
    val_eligible = val_full.loc[val_mask].copy()
    # Coerce string categoricals to pandas Categorical with a category set
    # shared across train + val + both test folds. Doing it once and
    # uniformly avoids two LightGBM gotchas:
    # 1. ``model.fit`` rejects object/string dtype outright.
    # 2. ``model.predict`` later fails with a misleading "train and valid
    #    dataset categorical_feature do not match" error if the predict
    #    input has different dtype (object vs categorical) than what the
    #    model stored at fit time.
    # We use the union cat_a ∪ cat_b so both Model A and Model B see
    # consistent dtypes throughout.
    union_cat_cols = sorted(set(cat_a) | set(cat_b))
    if union_cat_cols:
        train_eligible, val_eligible, folds = _coerce_categorical_union(
            train_eligible, val_eligible, folds, union_cat_cols
        )
    logger.info(
        "identical-rows guard: train %d -> %d (%.1f%% kept), val %d -> %d (%.1f%% kept)",
        len(train_full),
        len(train_eligible),
        100 * len(train_eligible) / max(len(train_full), 1),
        len(val_full),
        len(val_eligible),
        100 * len(val_eligible) / max(len(val_full), 1),
    )
    if train_eligible.empty:
        raise RuntimeError(
            "identical-rows guard left zero training rows - every train row has "
            "at least one null in the Model B column set. Check enrichment."
        )

    y_train = train_eligible[target]
    y_val = val_eligible[target]

    # ---- Fit -------------------------------------------------------------
    logger.info("fitting Model A (%d feature columns, no SA2 block)", len(cols_a))
    fit_a = fit_lgbm(
        X_train=train_eligible,
        y_train=y_train,
        X_val=val_eligible,
        y_val=y_val,
        feature_columns=cols_a,
        categorical_columns=cat_a,
    )
    logger.info("fitting Model B (%d feature columns, with SA2 block)", len(cols_b))
    fit_b = fit_lgbm(
        X_train=train_eligible,
        y_train=y_train,
        X_val=val_eligible,
        y_val=y_val,
        feature_columns=cols_b,
        categorical_columns=cat_b,
    )

    # ---- Persist ---------------------------------------------------------
    _save_pickle(fit_a.model, out_dir / "model_a.pkl")
    _save_pickle(fit_b.model, out_dir / "model_b.pkl")
    _save_feature_lists(out_dir / "feature_lists.json", fit_a, fit_b)

    if save_predictions:
        _save_predictions(folds, fit_a, fit_b, out_dir, target=target)

    logger.info("wrote models + audit to %s", out_dir)
    return {"A": fit_a, "B": fit_b}


# ---- internals -------------------------------------------------------------


def _warn_on_missing_blocks(df: pd.DataFrame, blocks: tuple[str, ...]) -> None:
    """Log a single WARNING enumerating any spec-defined feature columns
    that aren't in ``df``. Modeling continues without them — useful when
    ``build/make_features.py`` hasn't yet emitted a column the spec
    promises.
    """
    from fuel_pred.train.feature_blocks import BLOCK_COLUMNS

    expected: list[str] = []
    for b in blocks:
        expected.extend(BLOCK_COLUMNS[b])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning(
            "%d spec-defined feature column(s) absent from features.parquet "
            "(modeling will proceed without them): %s",
            len(missing),
            missing,
        )


def _coerce_categorical_union(
    train: pd.DataFrame,
    val: pd.DataFrame,
    folds: dict[str, pd.DataFrame],
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Cast ``columns`` to pandas Categorical with a category set shared
    across train + val + both test folds.

    Sharing the category set ensures:
    - LightGBM's val Dataset uses the same code-to-string mapping the
      train Dataset learned.
    - ``model.predict`` later doesn't crash on the test folds (which
      otherwise carry raw object dtypes the model wasn't fit against).

    Returns coerced (train, val, folds dict). Folds dict is replaced
    with copies so caller's original dict isn't mutated. No-ops on
    columns already typed as Categorical (re-casting would lose
    ordering).
    """
    out_train = train.copy()
    out_val = val.copy()
    out_folds = {name: df.copy() for name, df in folds.items()}

    # Build per-column categories from the union of all available frames.
    sources = [out_train, out_val, *out_folds.values()]
    for col in columns:
        if not all(col in src.columns for src in sources):
            continue
        if isinstance(out_train[col].dtype, pd.CategoricalDtype):
            continue
        union = pd.concat(
            [src[col] for src in sources], ignore_index=True
        ).dropna().unique()
        dtype = pd.CategoricalDtype(categories=pd.Index(union))
        out_train[col] = out_train[col].astype(dtype)
        out_val[col] = out_val[col].astype(dtype)
        for name in out_folds:
            out_folds[name][col] = out_folds[name][col].astype(dtype)
    return out_train, out_val, out_folds


def _save_pickle(obj: object, path: Path) -> None:
    """Pickle a model atomically (write tmp, rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh)
    tmp.replace(path)
    logger.info("wrote %s", path)


def _save_feature_lists(path: Path, fit_a: FitResult, fit_b: FitResult) -> None:
    """Serialise the feature lists + best-iteration audit trail.

    Lets the comparison report (Phase 8) and the explainability notebook
    (Phase 7 §9.3) recover exactly which columns each model used without
    re-loading the pickles.
    """
    payload = {
        "A": {
            "feature_columns": fit_a.feature_columns,
            "categorical_columns": fit_a.categorical_columns,
            "best_iteration": fit_a.best_iteration,
            "best_val_mae": fit_a.best_score,
        },
        "B": {
            "feature_columns": fit_b.feature_columns,
            "categorical_columns": fit_b.categorical_columns,
            "best_iteration": fit_b.best_iteration,
            "best_val_mae": fit_b.best_score,
        },
        "config": {
            # Snapshot the hyperparameters used so a future re-run can be
            # diffed against this one.
            "lgbm_params": {
                k: (v if not isinstance(v, type) else str(v))
                for k, v in config.LGBM_PARAMS.items()
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)


def _save_predictions(
    folds: dict[str, pd.DataFrame],
    fit_a: FitResult,
    fit_b: FitResult,
    out_dir: Path,
    *,
    target: str,
) -> None:
    """Write per-fold parquet with both models' predictions side-by-side.

    Schema: ``station_id, fuel_code, date, y_true, y_pred_a, y_pred_b``.
    This is what ``evaluate.compare`` consumes — keeps the eval pass fast
    and re-runnable without invoking LightGBM again.
    """
    for fold_name in ("test_normal", "test_crisis"):
        df = folds[fold_name]
        if df.empty:
            logger.warning("fold %s empty - skipping prediction parquet", fold_name)
            continue
        rows = pd.DataFrame(
            {
                "station_id": df["station_id"].to_numpy(),
                "fuel_code": df["fuel_code"].to_numpy(),
                "date": df["date"].to_numpy(),
                "y_true": df[target].to_numpy(),
                "y_pred_a": fit_a.model.predict(df[fit_a.feature_columns]),
                "y_pred_b": fit_b.model.predict(df[fit_b.feature_columns]),
            }
        )
        path = out_dir / f"predictions_{fold_name}.parquet"
        rows.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        logger.info("wrote %s (%d rows)", path, len(rows))


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
        help="skip writing per-fold prediction parquets (eval can still run "
        "via the pickles, just slower)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    train(
        args.features,
        args.out,
        target=args.target,
        save_predictions=not args.no_predictions,
    )


if __name__ == "__main__":
    main()
