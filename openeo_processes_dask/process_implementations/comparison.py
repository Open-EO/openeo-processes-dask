from typing import Callable, Optional

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from xarray.core.duck_array_ops import notnull

__all__ = [
    "is_infinite",
    "is_valid",
    "is_nodata",
    "is_nan",
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
]


def is_infinite(x: ArrayLike):
    if x is None:
        return False
    if isinstance(x, str) or x.dtype.kind.lower() in ["u", "s"]:
        return False
    return np.isinf(x)


def is_valid(x: ArrayLike):
    finite = np.logical_not(is_infinite(x))
    return np.logical_and(notnull(x), finite)


def is_nodata(x: ArrayLike):
    return np.logical_not(is_valid(x))


def is_nan(x: ArrayLike):
    return is_nodata(x)


def eq(
    x: ArrayLike,
    y: ArrayLike,
    delta: Optional[float] = None,
    case_sensitive: Optional[bool] = True,
):
    if is_nodata(x) or is_nodata(y):
        return np.nan
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
    if is_nodata(x) or is_nodata(y):
        return np.nan
    else:
        return np.logical_not(eq_val)


def gt(x: ArrayLike, y: ArrayLike):
    if is_nodata(x) or is_nodata(y):
        return np.nan
    gt_ar = x > y
    return gt_ar


def gte(x: ArrayLike, y: ArrayLike):
    if is_nodata(x) or is_nodata(y):
        return np.nan
    gte_ar = (x - y) >= 0
    return gte_ar


def lt(x: ArrayLike, y: ArrayLike):
    if is_nodata(x) or is_nodata(y):
        return np.nan
    lt_ar = x < y
    return lt_ar


def lte(x: ArrayLike, y: ArrayLike):
    if is_nodata(x) or is_nodata(y):
        return np.nan
    lte_ar = x <= y
    return lte_ar


def between(
    x: ArrayLike,
    min: float,
    max: float,
    exclude_max: Optional[bool] = False,
):
    if is_nodata(x) or is_nodata(min) or is_nodata(max):
        return np.nan
    if exclude_max:
        bet = np.logical_and(gte(x, y=min), lt(x, y=max))
    else:
        bet = np.logical_and(gte(x, y=min), lte(x, y=max))
    return bet
