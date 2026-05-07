"""Regression metrics for fuel-price prediction.

Wrapper functions that accept arrays / Series and return scalars. Kept simple
so they can be unit-tested cheaply.

Spec: spec.md §8.5.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    raise NotImplementedError


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    raise NotImplementedError


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """MAPE in percent."""
    raise NotImplementedError


def median_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    raise NotImplementedError


def quantile_absolute_error(y_true: ArrayLike, y_pred: ArrayLike, q: float = 0.9) -> float:
    """q-th percentile of the absolute errors."""
    raise NotImplementedError
