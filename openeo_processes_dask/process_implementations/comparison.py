from typing import Optional

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike
from xarray.core.duck_array_ops import notnull

from openeo_processes_dask.process_implementations.cubes.utils import _is_dask_array
from openeo_processes_dask.process_implementations.utils import get_scalar_type

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
    if np.issubsctype(get_scalar_type(x), np.number):
        return np.isinf(x)
    else:
        return False


def is_valid(x: ArrayLike):
    finite = np.logical_not(is_infinite(x))
    return np.logical_and(notnull(x), finite)


def eq(
    x: ArrayLike,
    y: ArrayLike,
    delta: Optional[float] = None,
    case_sensitive: Optional[bool] = True,
):
    x_dtype = get_scalar_type(x)
    y_dtype = get_scalar_type(y)

    if np.issubsctype(x_dtype, np.number) and np.issubsctype(y_dtype, np.number):
        if delta:
            ar_eq = np.isclose(x, y, atol=delta)
        else:
            ar_eq = x == y

    elif np.issubsctype(x_dtype, np.bool_) and np.issubsctype(y_dtype, np.bool_):
        ar_eq = x == y

    elif np.issubsctype(x_dtype, np.flexible) and np.issubsctype(y_dtype, np.flexible):
        if not case_sensitive:
            if np.issubsctype(get_scalar_type(x), np.character):
                x = np.char.lower(x)
            if np.issubsctype(get_scalar_type(y), np.character):
                y = np.char.lower(y)
        ar_eq = x == y

    elif np.issubsctype(x_dtype, np.flexible) and np.issubsctype(y_dtype, np.flexible):
        ar_eq = x == y
    else:
        return False

    if _is_dask_array(x):
        x_is_null = da.map_blocks(notnull, x)
    else:
        x_is_null = notnull(x)
    if _is_dask_array(y):
        y_is_null = da.map_blocks(notnull, y)
    else:
        y_is_null = notnull(y)

    null_mask = np.logical_and(x_is_null, y_is_null)

    if _is_dask_array(ar_eq):
        result = da.where(null_mask, ar_eq, np.nan)
    else:
        result = np.where(null_mask, ar_eq, np.nan)
    return result


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
