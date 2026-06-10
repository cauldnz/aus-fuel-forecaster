"""Fit Models A (no SA2), B (with SA2), and B' (B + venue) on the feature matrix.

All three models use identical hyperparameters (``config.LGBM_PARAMS``)
and identical training rows: only rows where every SA2 column is non-null
are used. This prevents the augmentor from looking better just because
its richer column set excluded harder examples. Venue columns can be null
on a small set of rows and LightGBM handles them natively, so they are
NOT subject to the row-filter.

Model B' (spec §13.6 Phase 1) adds the VENUE feature block — 4 static
nearest-venue features + ``cal_is_pre_long_weekend`` — to test whether
they carry signal beyond what Model B already extracts from
``stn_is_metro`` and other existing features.

Splits per spec.md §8.3 (delegated to ``train.folds.split_folds``).

Outputs (under ``out_dir``, typically ``models/``):
    model_a.pkl                          # pickled LGBMRegressor
    model_b.pkl                          # pickled LGBMRegressor
    model_b_prime.pkl                    # pickled LGBMRegressor (Phase 1 ablation)
    feature_lists.json                   # column lists per model + audit
    predictions_test_normal.parquet      # all three models' preds (y_pred_a/b/b_prime)
    predictions_test_crisis.parquet      # all three models' preds (y_pred_a/b/b_prime)
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from fuel_pred import config
from fuel_pred.train._fit import DEFAULT_LOG_PERIOD, FitResult, fit_lgbm
from fuel_pred.train.feature_blocks import (
    BLOCK_COLUMNS,
    MODEL_A_BLOCKS,
    MODEL_A_GFS_BLOCKS,
    MODEL_B_BLOCKS,
    MODEL_B_GFS_BLOCKS,
    MODEL_B_PRIME_BLOCKS,
    categorical_columns,
    feature_columns,
)
from fuel_pred.train.folds import FoldConfig, split_folds

logger = logging.getLogger(__name__)

TARGET_COLUMN: str = "y_t1"


def _load_and_filter_target(features_path: Path, target: str) -> pd.DataFrame:
    """Load features.parquet and filter to U91 + non-null target.

    Extracted as a shared helper so both ``train()`` (single-split) and
    ``train.cv.train_kfold()`` load the panel the same way.
    """
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
    return work


def train(
    features_path: Path,
    out_dir: Path,
    *,
    fold: FoldConfig | None = None,
    target: str = TARGET_COLUMN,
    save_predictions: bool = True,
    log_period: int = DEFAULT_LOG_PERIOD,
    n_estimators: int | None = None,
) -> dict[str, FitResult]:
    """Fit Models A and B; persist artefacts under ``out_dir``.

    **v2.x single-split path** (spec §8.3 historical). For the v3.0
    k-fold path see ``fuel_pred.train.cv.train_kfold``.

    Args:
        features_path: ``data/processed/features.parquet`` from
            ``build.make_features``.
        out_dir: typically ``models/``. Created if missing.
        fold: optional override of the spec §8.3 fold boundaries
            (tests pass a synthetic FoldConfig).
        target: target column name; default ``y_t1`` per spec §7.8.
            ``y_t1_t7`` is also valid for the longer-horizon variant.
        log_period: emit a per-iteration eval line every ``log_period``
            boosting rounds (passed to ``lgb.log_evaluation``). Default
            50 — gives ~30-40 lines per model at the spec's 2000-iter
            ceiling. Set to 0 to silence (e.g. in test runs that
            capture stdout).
        n_estimators: cap on boosting rounds. When None (default), uses
            the spec §8.2 value from ``config.LGBM_PARAMS`` (2000).
            Set lower (e.g. 800) for rough-iteration runs where the
            last few % of training gain isn't worth the wall clock.
            Both models receive the same value — keeps the A/B
            comparison apples-to-apples per spec §8.4.
        save_predictions: if True (default), also writes per-fold
            prediction parquets so the comparison-report writer
            (Phase 8) doesn't need to re-load the models.

    Returns:
        ``{"A": FitResult, "B": FitResult, "B_PRIME": FitResult}``
        for downstream callers that want the in-memory models.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    work = _load_and_filter_target(features_path, target)

    # ---- Split into the four time-based folds ----------------------------
    folds = split_folds(work, fold=fold)

    # Hand the four-fold v2.x layout to the shared per-fold trainer.
    return _train_one_fold(
        train_full=folds["train"],
        val_full=folds["val"],
        test_folds={
            "test_normal": folds["test_normal"],
            "test_crisis": folds["test_crisis"],
        },
        out_dir=out_dir,
        target=target,
        save_predictions=save_predictions,
        log_period=log_period,
        n_estimators=n_estimators,
    )


def _train_one_fold(
    *,
    train_full: pd.DataFrame,
    val_full: pd.DataFrame,
    test_folds: dict[str, pd.DataFrame],
    out_dir: Path,
    target: str = TARGET_COLUMN,
    save_predictions: bool = True,
    log_period: int = DEFAULT_LOG_PERIOD,
    n_estimators: int | None = None,
    random_state: int | None = None,
    models_to_fit: tuple[str, ...] = ("A", "B", "B_PRIME"),
) -> dict[str, FitResult]:
    """Fit A/B/B' on one fold's (train, val); predict each entry in test_folds.

    Shared by the v2.x ``train()`` path (called once with the four-fold
    layout) and the v3.0 ``train.cv.train_kfold()`` path (called per CV
    fold). Each call writes its A/B/B' pickles + ``feature_lists.json``
    to ``out_dir``, optionally + ``predictions_<test_fold_name>.parquet``
    for each entry in ``test_folds``.

    Args:
        train_full: train slice (pre-identical-rows-guard).
        val_full: val slice for early stopping (pre-identical-rows-guard).
        test_folds: ``{fold_name: test_df}``. For v2.x:
            ``{"test_normal": ..., "test_crisis": ...}``. For k-fold:
            ``{"test": ...}``. Empty dicts are valid — no test
            predictions get written.
        out_dir: persistence target. Created if missing.
        target / save_predictions / log_period / n_estimators: see
            ``train()`` docstring.
        random_state: optional override of
            ``config.LGBM_PARAMS["random_state"]`` (default 42). Used by
            seed-noise experiments (v3.0 Phase 3 next-step #2) to estimate
            LightGBM's seed-driven variance floor. When None, falls back
            to the spec default.
        models_to_fit: subset of ``("A", "B", "B_PRIME")`` to actually
            train + persist. Defaults to all three (production behaviour).
            Pass ``("A",)`` for seed-noise experiments that only need
            Model A — skips ~2/3 of the per-fold compute.

    Returns:
        Dict keyed by the entries of ``models_to_fit`` — typically
        ``{"A": FitResult, "B": FitResult, "B_PRIME": FitResult}``.
    """
    valid_ids = {"A", "B", "B_PRIME"}
    bad = [m for m in models_to_fit if m not in valid_ids]
    if bad:
        raise ValueError(
            f"unknown model id(s) in models_to_fit={models_to_fit!r}: {bad}. "
            f"Allowed: {sorted(valid_ids)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # `work` for column-presence checks: the union of train + val + tests.
    # Sufficient for `feature_columns(strict=False)` + spec-drift WARNING.
    column_audit_src = train_full

    # ---- Pick feature columns per model variant --------------------------
    # Lax mode: warn if the spec promises a column that build/make_features
    # doesn't actually emit, but proceed with whatever's there. Strict mode
    # is intended for callers (notebooks / interactive use) that want the
    # spec drift to surface as a hard error; the production training
    # pipeline should be defensive about known-pending feature columns.
    #
    # Weather block selection (spec §13.7 v2.0): when WEATHER_SOURCE resolves
    # to 'gfs', swap the Open-Meteo "wx" block for the GFS-day-1 "wx_gfs"
    # block. Everything else stays the same. Model B' (the venue+long-weekend
    # ablation) keeps the Open-Meteo block — it was a 2026-05 experiment
    # against the v1 baseline; not worth re-running against v2 weather.
    weather_source = config.resolve_weather_source()
    if weather_source == "gfs":
        a_blocks = MODEL_A_GFS_BLOCKS
        b_blocks = MODEL_B_GFS_BLOCKS
        logger.info("weather source: gfs — using MODEL_A_GFS_BLOCKS / MODEL_B_GFS_BLOCKS")
    else:
        a_blocks = MODEL_A_BLOCKS
        b_blocks = MODEL_B_BLOCKS
        logger.info("weather source: openmeteo — using canonical MODEL_A_BLOCKS / MODEL_B_BLOCKS")

    _warn_on_missing_blocks(column_audit_src, MODEL_B_PRIME_BLOCKS)
    cols_a = feature_columns(column_audit_src, a_blocks, strict=False)
    cols_b = feature_columns(column_audit_src, b_blocks, strict=False)
    cols_b_prime = feature_columns(column_audit_src, MODEL_B_PRIME_BLOCKS, strict=False)
    cat_a = categorical_columns(cols_a)
    cat_b = categorical_columns(cols_b)
    cat_b_prime = categorical_columns(cols_b_prime)
    logger.info(
        "feature counts: Model A = %d (%d cat) ; "
        "Model B = %d (%d cat) ; Model B' = %d (%d cat)",
        len(cols_a),
        len(cat_a),
        len(cols_b),
        len(cat_b),
        len(cols_b_prime),
        len(cat_b_prime),
    )

    # Rebuild a folds-like dict so _coerce_* still work (they expect one)
    folds: dict[str, pd.DataFrame] = dict(test_folds)

    # NOTE: the rest of this function is now the body of
    # `_train_one_fold` — the docstring above documents the inputs. The
    # `folds` dict has been renamed to `test_folds` semantically but kept
    # as `folds` for the categorical/object coercion helpers' signature.

    # ---- Identical-rows guard (spec §8.4) --------------------------------
    # Both models train on rows where every column in the SA2 block is
    # non-null. Spec §8.4 originally read "every column required by Model
    # B" but the intent — confirmed by spec §8.4's own gloss — is that
    # the SA2 join shouldn't bias the comparison. Other naturally-sparse
    # columns (xfuel_dl_*, upstream_tgp_*, occasional Tier-2 macros) are
    # in BOTH models' feature sets and LightGBM handles their nulls
    # natively. Filtering on every Model B column is over-strict and on
    # real corpora can leave zero training rows because rare-coverage
    # columns combine multiplicatively.
    #
    # The right test: keep rows whose SA2 block is fully populated. A and
    # B see identical row sets, so Model B's only structural advantage is
    # the SA2 columns themselves. That's exactly what the §8.4
    # "apples-to-apples" comparison is supposed to isolate.
    sa2_cols = list(BLOCK_COLUMNS["sa2"])
    sa2_cols_present = [c for c in sa2_cols if c in train_full.columns]
    train_mask = train_full[sa2_cols_present].notna().all(axis=1)
    val_mask = val_full[sa2_cols_present].notna().all(axis=1)
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
    # We use the union cat_a ∪ cat_b ∪ cat_b_prime so every model sees
    # consistent dtypes throughout.
    union_cat_cols = sorted(set(cat_a) | set(cat_b) | set(cat_b_prime))
    if union_cat_cols:
        train_eligible, val_eligible, folds = _coerce_categorical_union(
            train_eligible, val_eligible, folds, union_cat_cols
        )
    # Defensive: coerce any remaining object-dtype feature columns to
    # numeric. LightGBM rejects object dtype outright. Two ways a column
    # ends up object in features.parquet:
    #   - It was 100% null at write time (e.g. upstream_tgp_*,
    #     ctx_cash_rate during fold periods where the fetcher had no data)
    #     — pandas keeps the previous object inference rather than
    #     promoting to float.
    #   - The build step had mixed types (numeric + None) — pandas falls
    #     back to object when it can't unify them.
    # Both cases are make_features.py bugs we should fix at the source,
    # but the coercion here unblocks training on existing features.parquet.
    # Tracked separately as an issue.
    non_cat_feature_cols = [c for c in cols_b_prime if c not in union_cat_cols]
    train_eligible, val_eligible, folds = _coerce_object_to_numeric(
        train_eligible, val_eligible, folds, non_cat_feature_cols
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
            "at least one null in the SA2 column set. Check enrichment "
            f"({len(sa2_cols_present)} sa2_* columns checked: {sa2_cols_present})."
        )

    y_train = train_eligible[target]
    y_val = val_eligible[target]

    # ---- Build per-fit params (with optional n_estimators override) -----
    # Both A and B get the same params snapshot — keeps the comparison
    # apples-to-apples per spec §8.4. Override applied here rather than
    # mutating config.LGBM_PARAMS so the global stays clean.
    fit_params = dict(config.LGBM_PARAMS)
    if n_estimators is not None:
        spec_default = config.LGBM_PARAMS.get("n_estimators")
        fit_params["n_estimators"] = n_estimators
        logger.info(
            "n_estimators override: %d (spec default %s) — early stopping "
            "still fires at early_stopping_rounds=%s",
            n_estimators,
            spec_default,
            config.LGBM_PARAMS.get("early_stopping_rounds"),
        )
    if random_state is not None:
        spec_default_rs = config.LGBM_PARAMS.get("random_state")
        fit_params["random_state"] = random_state
        logger.info(
            "random_state override: %d (spec default %s) — used for "
            "seed-noise experiments (v3.0 Phase 3 next-step #2)",
            random_state, spec_default_rs,
        )

    # ---- Fit -------------------------------------------------------------
    # Conditional fits driven by ``models_to_fit`` — lets the seed-noise
    # experiment (v3.0 Phase 3 #2) train only Model A per seed-run, cutting
    # wall-clock by ~2/3.
    fits: dict[str, FitResult] = {}
    if "A" in models_to_fit:
        logger.info("fitting Model A (%d feature columns, no SA2 block)", len(cols_a))
        fits["A"] = fit_lgbm(
            X_train=train_eligible,
            y_train=y_train,
            X_val=val_eligible,
            y_val=y_val,
            feature_columns=cols_a,
            categorical_columns=cat_a,
            params=fit_params,
            log_period=log_period,
        )
    if "B" in models_to_fit:
        logger.info("fitting Model B (%d feature columns, with SA2 block)", len(cols_b))
        fits["B"] = fit_lgbm(
            X_train=train_eligible,
            y_train=y_train,
            X_val=val_eligible,
            y_val=y_val,
            feature_columns=cols_b,
            categorical_columns=cat_b,
            params=fit_params,
            log_period=log_period,
        )
    if "B_PRIME" in models_to_fit:
        logger.info(
            "fitting Model B' (%d feature columns, B + venue block — spec §13.6 Phase 1)",
            len(cols_b_prime),
        )
        fits["B_PRIME"] = fit_lgbm(
            X_train=train_eligible,
            y_train=y_train,
            X_val=val_eligible,
            y_val=y_val,
            feature_columns=cols_b_prime,
            categorical_columns=cat_b_prime,
            params=fit_params,
            log_period=log_period,
        )

    # ---- Persist ---------------------------------------------------------
    if "A" in fits:
        _save_pickle(fits["A"].model, out_dir / "model_a.pkl")
    if "B" in fits:
        _save_pickle(fits["B"].model, out_dir / "model_b.pkl")
    if "B_PRIME" in fits:
        _save_pickle(fits["B_PRIME"].model, out_dir / "model_b_prime.pkl")
    _save_feature_lists(
        out_dir / "feature_lists.json",
        fits.get("A"),
        fits.get("B"),
        fits.get("B_PRIME"),
    )

    if save_predictions:
        _save_predictions(
            folds,
            fits.get("A"),
            fits.get("B"),
            fits.get("B_PRIME"),
            out_dir,
            target=target,
        )

    logger.info("wrote models + audit to %s", out_dir)
    return fits


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


def _coerce_object_to_numeric(
    train: pd.DataFrame,
    val: pd.DataFrame,
    folds: dict[str, pd.DataFrame],
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Cast any object-dtype columns in ``columns`` to numeric across all frames.

    LightGBM only accepts int / float / bool / Categorical dtypes; object
    columns raise. ``pd.to_numeric(errors='coerce')`` turns
    actual-numeric values into floats and any non-numeric leftovers into
    NaN — which LightGBM handles natively.

    Logs which columns were coerced (one INFO line) so the surface area
    is visible. Real fix belongs in build/make_features.py; tracked
    as a separate issue.
    """
    out_train = train.copy()
    out_val = val.copy()
    out_folds = {name: df.copy() for name, df in folds.items()}

    coerced: list[str] = []
    for col in columns:
        if col not in out_train.columns:
            continue
        if out_train[col].dtype != object:
            continue
        coerced.append(col)
        out_train[col] = pd.to_numeric(out_train[col], errors="coerce")
        if col in out_val.columns:
            out_val[col] = pd.to_numeric(out_val[col], errors="coerce")
        for name in out_folds:
            if col in out_folds[name].columns:
                out_folds[name][col] = pd.to_numeric(
                    out_folds[name][col], errors="coerce"
                )

    if coerced:
        logger.info(
            "coerced %d object-dtype feature column(s) to numeric "
            "(make_features.py bug; tracked separately): %s",
            len(coerced),
            coerced,
        )
    return out_train, out_val, out_folds


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


def _save_feature_lists(
    path: Path,
    fit_a: FitResult | None,
    fit_b: FitResult | None,
    fit_b_prime: FitResult | None,
) -> None:
    """Serialise the feature lists + best-iteration audit trail.

    Lets the comparison report (Phase 8) and the explainability notebook
    (Phase 7 §9.3) recover exactly which columns each model used without
    re-loading the pickles. Includes Model B' (spec §13.6 Phase 1) under
    the ``"B_PRIME"`` key alongside ``"A"`` and ``"B"``.

    Any of the three fits may be ``None`` — the caller passed a partial
    ``models_to_fit`` subset (e.g. the seed-noise experiment trains only
    Model A). Omitted models are simply absent from the audit payload.
    """
    def _serialise(fit: FitResult) -> dict[str, object]:
        return {
            "feature_columns": fit.feature_columns,
            "categorical_columns": fit.categorical_columns,
            "best_iteration": fit.best_iteration,
            "best_val_mae": fit.best_score,
            # Per-feature importances (gain + split). Lets the
            # comparison report and the explainability notebook rank
            # features without re-loading the pickle.
            "importance_gain": fit.importance_gain,
            "importance_split": fit.importance_split,
        }

    payload: dict[str, object] = {}
    if fit_a is not None:
        payload["A"] = _serialise(fit_a)
    if fit_b is not None:
        payload["B"] = _serialise(fit_b)
    if fit_b_prime is not None:
        payload["B_PRIME"] = _serialise(fit_b_prime)
    payload["config"] = {
        # Snapshot the hyperparameters used so a future re-run can be
        # diffed against this one.
        "lgbm_params": {
            k: (v if not isinstance(v, type) else str(v))
            for k, v in config.LGBM_PARAMS.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)


def _save_predictions(
    folds: dict[str, pd.DataFrame],
    fit_a: FitResult | None,
    fit_b: FitResult | None,
    fit_b_prime: FitResult | None,
    out_dir: Path,
    *,
    target: str,
) -> None:
    """Write per-fold parquet with all three models' predictions side-by-side.

    Iterates over every key in ``folds`` and writes
    ``predictions_<fold_name>.parquet``. v2.x callers pass
    ``{"test_normal": ..., "test_crisis": ...}``; v3.0 k-fold callers
    pass ``{"test": ...}``.

    Schema: ``station_id, fuel_code, date, y_true, y_pred_a, y_pred_b,
    y_pred_b_prime``. This is what ``evaluate.compare`` (single-split)
    and ``evaluate.compare_kfold`` (k-fold) consume — keeps eval fast
    and re-runnable without invoking LightGBM again.

    Any of the three fits may be ``None`` (the caller used a partial
    ``models_to_fit`` subset). Omitted models' ``y_pred_*`` columns are
    simply absent from the output parquet.
    """
    for fold_name, df in folds.items():
        if df.empty:
            logger.warning("fold %s empty - skipping prediction parquet", fold_name)
            continue
        cols: dict[str, object] = {
            "station_id": df["station_id"].to_numpy(),
            "fuel_code": df["fuel_code"].to_numpy(),
            "date": df["date"].to_numpy(),
            "y_true": df[target].to_numpy(),
        }
        if fit_a is not None:
            cols["y_pred_a"] = fit_a.model.predict(df[fit_a.feature_columns])
        if fit_b is not None:
            cols["y_pred_b"] = fit_b.model.predict(df[fit_b.feature_columns])
        if fit_b_prime is not None:
            cols["y_pred_b_prime"] = fit_b_prime.model.predict(
                df[fit_b_prime.feature_columns]
            )
        rows = pd.DataFrame(cols)
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
    parser.add_argument(
        "--log-period",
        type=int,
        default=DEFAULT_LOG_PERIOD,
        help=(
            "emit a per-iteration eval line every N boosting rounds "
            "(default %(default)s; ~30-40 lines per model at the spec's "
            "2000-iter ceiling). Set 0 to silence, or 1 for every-iter "
            "output (XGBoost-style)"
        ),
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help=(
            "cap on boosting rounds (default: spec §8.2 value, currently "
            "2000). Lower for rough-iteration runs where the last few %% "
            "of training gain isn't worth the wall clock — e.g. "
            "--n-estimators 800. Both Model A and Model B receive the "
            "same value to keep the §8.4 comparison apples-to-apples"
        ),
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
        log_period=args.log_period,
        n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
