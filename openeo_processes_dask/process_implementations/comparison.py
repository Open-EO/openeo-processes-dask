from typing import Callable, Optional

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

__all__ = [
    "is_nodata",
    "is_nan",
    "is_valid",
    "is_infinite",
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
]


def is_nodata(x: ArrayLike):
    return x is None


def is_nan(x: ArrayLike):
    return np.isnan(x)


def is_valid(x: ArrayLike):
    if x is None:
        return False
    return np.logical_not(np.logical_or(np.isnan(x), np.isinf(x)))


def is_infinite(x: ArrayLike):
    if x is None:
        return False
    return np.isinf(x)


def eq(
    x: ArrayLike,
    y: ArrayLike,
    delta: Optional[float] = None,
    case_sensitive: Optional[bool] = True,
):
    if x is None or y is None:
        return None
    if x is False or y is False:
        return False
    if delta:
        ar_eq = np.isclose(x, y, atol=delta)
    elif not case_sensitive:
        ar_eq = np.char.lower(x) == np.char.lower(y)
    else:
        ar_eq = x == y
    return ar_eq


def neq(
    x: ArrayLike,
    y: ArrayLike,
    delta: Optional[float] = None,
    case_sensitive: Optional[bool] = True,
):
    eq_val = eq(x, y, delta=delta, case_sensitive=case_sensitive)
    if eq_val is None:
        return None
    else:
        return np.logical_not(eq_val)


def gt(x: ArrayLike, y: ArrayLike):
    if x is None or y is None:
        return None
    gt_ar = x > y
    return gt_ar


def gte(x: ArrayLike, y: ArrayLike):
    if x is None or y is None:
        return None
    gte_ar = (x - y) >= 0
    return gte_ar


def lt(x: ArrayLike, y: ArrayLike):
    if x is None or y is None:
        return None
    lt_ar = x < y
    return lt_ar


def lte(x: ArrayLike, y: ArrayLike):
    if x is None or y is None:
        return None
    lte_ar = x <= y
    return lte_ar


def between(
    x: ArrayLike,
    min: float,
    max: float,
    exclude_max: Optional[bool] = False,
):
    if x is None or min is None or max is None:
        return None
    if exclude_max:
        bet = np.logical_and(gte(x, y=min), lt(x, y=max))
    else:
        bet = np.logical_and(gte(x, y=min), lte(x, y=max))
    return bet
