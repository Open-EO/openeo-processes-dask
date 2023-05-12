from typing import Callable, Optional

import dask.array as da
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
    if (
        type(x) in [str, list, dict]
        or type(x) in [np.ndarray, da.core.Array]
        and x.dtype.kind.lower() in ["u", "s", "o"]
    ):
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
    if not type(x) in [np.ndarray, da.core.Array] and not type(y) in [
        np.ndarray,
        da.core.Array,
    ]:
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
    return np.where(np.logical_and(is_valid(x), is_valid(y)), ar_eq, np.nan)


def neq(
    x: ArrayLike,
    y: ArrayLike,
    delta: Optional[float] = None,
    case_sensitive: Optional[bool] = True,
):
    eq_val = eq(x, y, delta=delta, case_sensitive=case_sensitive)
    return np.where(
        np.logical_and(is_valid(x), is_valid(y)), np.logical_not(eq_val), np.nan
    )


def gt(x: ArrayLike, y: ArrayLike):
    if not type(x) in [np.ndarray, da.core.Array] and not type(y) in [
        np.ndarray,
        da.core.Array,
    ]:
        if is_nodata(x) or is_nodata(y):
            return np.nan
    gt_ar = x > y
    return np.where(np.logical_and(is_valid(x), is_valid(y)), gt_ar, np.nan)


def gte(x: ArrayLike, y: ArrayLike):
    if not type(x) in [np.ndarray, da.core.Array] and not type(y) in [
        np.ndarray,
        da.core.Array,
    ]:
        if is_nodata(x) or is_nodata(y):
            return np.nan
    gte_ar = (x - y) >= 0
    return np.where(np.logical_and(is_valid(x), is_valid(y)), gte_ar, np.nan)


def lt(x: ArrayLike, y: ArrayLike):
    if not type(x) in [np.ndarray, da.core.Array] and not type(y) in [
        np.ndarray,
        da.core.Array,
    ]:
        if is_nodata(x) or is_nodata(y):
            return np.nan
    lt_ar = x < y
    return np.where(np.logical_and(is_valid(x), is_valid(y)), lt_ar, np.nan)


def lte(x: ArrayLike, y: ArrayLike):
    if not type(x) in [np.ndarray, da.core.Array] and not type(y) in [
        np.ndarray,
        da.core.Array,
    ]:
        if is_nodata(x) or is_nodata(y):
            return np.nan
    lte_ar = x <= y
    return np.where(np.logical_and(is_valid(x), is_valid(y)), lte_ar, np.nan)


def between(
    x: ArrayLike,
    min: float,
    max: float,
    exclude_max: Optional[bool] = False,
):
    if is_nodata(min) or is_nodata(max):
        return np.nan
    if exclude_max:
        bet = np.logical_and(gte(x, y=min), lt(x, y=max))
    else:
        bet = np.logical_and(gte(x, y=min), lte(x, y=max))
    return np.where(is_valid(x), bet, np.nan)
