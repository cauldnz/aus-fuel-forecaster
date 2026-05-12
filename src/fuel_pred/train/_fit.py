"""Inner LightGBM training function — kept separate from the orchestrator.

Splitting this out makes the orchestrator easy to read (load → split →
guard → fit A → fit B → save) and keeps the LightGBM-specific knobs in
one place. Tests can also exercise the fit path with synthetic features
without going through the parquet I/O.

Spec: spec.md §8.1 (LightGBM), §8.2 (fixed hyperparameters).
"""
# X_train / X_val are sklearn-conventional names (uppercase X for the
# 2-D feature matrix vs. lowercase y for the 1-D target). Suppress the
# pep8-naming rule for this module rather than warring with convention.
# ruff: noqa: N803
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import lightgbm as lgb
import pandas as pd

from fuel_pred import config

logger = logging.getLogger(__name__)

# Route LightGBM's internal logger (the one `lgb.log_evaluation`,
# `lgb.early_stopping`, build warnings etc. write to) through our Python
# `logging` setup so its output inherits the timestamp formatter we
# configure in the CLI's `logging.basicConfig`. Without this, the
# `[50] valid_0's l1: 5.234` lines come out raw — useful but missing the
# `2026-05-12 06:46:26,470 ...` prefix.
#
# `register_logger` is process-global; calling it once at module import
# is fine because every consumer of fuel_pred.train benefits.
lgb.register_logger(logger.getChild("lightgbm"))


@dataclass(frozen=True)
class FitResult:
    """Output of ``fit_lgbm``: the trained model + a small audit trail.

    The audit trail is what we serialise into ``models/feature_lists.json``
    so the comparison report can recover the exact feature set without
    cracking open the pickle.
    """

    model: lgb.LGBMRegressor
    feature_columns: list[str]
    categorical_columns: list[str]
    best_iteration: int | None
    best_score: float | None  # validation MAE at best_iteration


DEFAULT_LOG_PERIOD: int = 50


def fit_lgbm(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_columns: list[str],
    categorical_columns: list[str] | None = None,
    params: dict[str, object] | None = None,
    log_period: int = DEFAULT_LOG_PERIOD,
) -> FitResult:
    """Fit one LightGBM regressor with early stopping on the val fold.

    Args:
        X_train, y_train: training features + target.
        X_val, y_val: validation features + target. Used by LightGBM
            for early stopping (rounds-without-improvement counter).
        feature_columns: explicit column list. Caller is responsible for
            choosing this — see ``train.feature_blocks.feature_columns``.
        categorical_columns: subset of ``feature_columns`` to mark as
            LightGBM-categorical. None ⇒ no categoricals (all numeric).
        params: hyperparameters dict; defaults to ``config.LGBM_PARAMS``
            (spec §8.2). Override only for tests.
        log_period: emit a per-iteration eval line every ``log_period``
            boosting rounds via ``lgb.log_evaluation``. ~30-40 lines per
            model at the spec's 2000-iteration ceiling. Set to 0 to
            silence (e.g. in tests where pytest captures stdout and the
            noise is unhelpful).

    Returns:
        ``FitResult`` with the model + audit trail.

    Notes:
        - Uses ``lgb.early_stopping`` callback rather than the deprecated
          ``early_stopping_rounds=`` constructor arg, so the parameter
          dict can stay declarative.
        - Categorical columns must be passed via ``categorical_feature=``
          on ``fit()`` (not on the constructor) so they apply to both
          train and val Datasets.
        - LightGBM accepts pandas Categorical / object dtypes natively
          — no manual encoding needed.
    """
    params = dict(params or config.LGBM_PARAMS)

    # Strip early_stopping_rounds from params; we wire it via a callback so
    # newer LightGBM versions don't deprecation-warn on every fit.
    # The cast is needed because dict.pop returns object; the actual values
    # in LGBM_PARAMS are int/float/str/bool — we know early_stopping_rounds
    # is int specifically.
    early_stopping_rounds = int(cast(int, params.pop("early_stopping_rounds", 0)))

    # n_estimators is an LGBMRegressor constructor arg, not a fit() kwarg.
    n_estimators = int(cast(int, params.pop("n_estimators", 1000)))

    # ``params`` is dict[str, object] — typed loosely on purpose because
    # LGBMRegressor accepts a wide variety of value types (str / int /
    # float / bool / None) and tying it to a TypedDict is more friction
    # than value. Cast to Any so mypy's per-kwarg validation against the
    # constructor signature doesn't fight the dict-unpacking.
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **cast(dict[str, Any], params))

    callbacks: list[Callable[..., Any]] = []
    if early_stopping_rounds > 0:
        callbacks.append(
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        )
    if log_period > 0:
        # ``lgb.log_evaluation(period=N)`` prints one line every N rounds:
        #   ``[N]    valid_0's l1: X.XXXXX``
        # Useful for "is training still alive / converging" visibility.
        # Goes to LightGBM's own logger (which writes to stderr by default);
        # keeps it separate from our INFO-level logger.
        callbacks.append(lgb.log_evaluation(period=log_period))

    # LightGBM accepts categoricals as pandas Categorical dtype or int codes
    # but rejects object/string dtype. The orchestrator
    # (``train_models._coerce_categorical_union``) is responsible for casting
    # them ahead of time so train + val share a category set across BOTH
    # Model A and Model B fits. We trust the caller here — assert defensively
    # so a bad caller fails loudly rather than producing a confusing
    # LightGBM error.
    if categorical_columns:
        for col in categorical_columns:
            if col not in X_train.columns:
                continue
            dtype = X_train[col].dtype
            if not isinstance(dtype, pd.CategoricalDtype):
                raise TypeError(
                    f"categorical column {col!r} must be pandas Categorical "
                    f"dtype before reaching fit_lgbm; got {dtype}. Cast it "
                    "via train_models._coerce_categorical_union."
                )

    # Direct kwargs (rather than dict-unpacking) so mypy can verify each
    # argument's type against LGBMRegressor.fit's signature.
    model.fit(
        X=X_train[feature_columns],
        y=y_train,
        eval_set=[(X_val[feature_columns], y_val)],
        categorical_feature=(categorical_columns if categorical_columns else "auto"),
        callbacks=(callbacks if callbacks else None),
    )

    best_iter = getattr(model, "best_iteration_", None)
    best_score: float | None = None
    if hasattr(model, "best_score_") and model.best_score_:
        try:
            # best_score_ is {eval_set_name: {metric_name: value}}; we have
            # one eval set, one metric. Pull the first value defensively.
            outer = next(iter(model.best_score_.values()))
            best_score = float(next(iter(outer.values())))
        except (StopIteration, TypeError, ValueError):
            best_score = None

    logger.info(
        "fit_lgbm: %d features (%d categorical), best_iteration=%s, "
        "best_val_mae=%s",
        len(feature_columns),
        len(categorical_columns or []),
        best_iter,
        f"{best_score:.4f}" if best_score is not None else "n/a",
    )

    return FitResult(
        model=model,
        feature_columns=list(feature_columns),
        categorical_columns=list(categorical_columns or []),
        best_iteration=best_iter,
        best_score=best_score,
    )
