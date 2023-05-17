from typing import Callable, Optional

import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from xarray.core.duck_array_ops import notnull

from openeo_processes_dask.process_implementations.utils import check_type

__all__ = [
    "is_infinite",
    "is_valid",
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
    if check_type(x, "str") or check_type(x, "list") or check_type(x, "dict"):
        return False
    return np.isinf(x)


def is_valid(x: ArrayLike):
    finite = np.logical_not(is_infinite(x))
    return np.logical_and(notnull(x), finite)


def eq(
    x: ArrayLike,
    y: ArrayLike,
    delta: Optional[float] = None,
    case_sensitive: Optional[bool] = True,
):
    if (
        check_type(x, "bool")
        and not check_type(y, "bool")
        or check_type(y, "bool")
        and not check_type(x, "bool")
    ):
        return False
    if delta and check_type(x, "float") and check_type(y, "float"):
        ar_eq = np.isclose(x, y, atol=delta)
    elif not case_sensitive and check_type(x, "str") and check_type(y, "str"):
        ar_eq = np.char.lower(x) == np.char.lower(y)
    else:
        ar_eq = x == y
    ar_eq = da.where(np.logical_and(notnull(x), notnull(y)), ar_eq, np.nan)
    return ar_eq


def neq(
    x: ArrayLike,
    y: ArrayLike,
    delta: Optional[float] = None,
    case_sensitive: Optional[bool] = True,
):
    eq_val = eq(x, y, delta=delta, case_sensitive=case_sensitive)
    return da.where(
        np.logical_and(notnull(x), notnull(y)), np.logical_not(eq_val), np.nan
    )


def gt(x: ArrayLike, y: ArrayLike):
    gt_ar = x > y
    return da.where(np.logical_and(notnull(x), notnull(y)), gt_ar, np.nan)


def gte(x: ArrayLike, y: ArrayLike):
    gte_ar = (x - y) >= 0
    return da.where(np.logical_and(notnull(x), notnull(y)), gte_ar, np.nan)


def lt(x: ArrayLike, y: ArrayLike):
    lt_ar = x < y
    return da.where(np.logical_and(notnull(x), notnull(y)), lt_ar, np.nan)


def lte(x: ArrayLike, y: ArrayLike):
    lte_ar = x <= y
    return da.where(np.logical_and(notnull(x), notnull(y)), lte_ar, np.nan)


def between(
    x: ArrayLike,
    min: float,
    max: float,
    exclude_max: Optional[bool] = False,
):
    if not notnull(min) or not notnull(max):
        return np.nan
    if exclude_max:
        bet = np.logical_and(gte(x, y=min), lt(x, y=max))
    else:
        bet = np.logical_and(gte(x, y=min), lte(x, y=max))
    return da.where(notnull(x), bet, np.nan)
