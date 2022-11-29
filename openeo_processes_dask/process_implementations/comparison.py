import math

import dask.array as da
import numpy as np
import xarray as xr

__all__ = [
    "is_empty",
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


def keep_attrs(x, y, data):
    if isinstance(x, xr.DataArray) and isinstance(y, xr.DataArray):
        for a in x.attrs:
            if a in y.attrs and (x.attrs[a] == y.attrs[a]):
                data.attrs[a] = x.attrs[a]
    elif isinstance(x, xr.DataArray):
        data.attrs = x.attrs
    elif isinstance(y, xr.DataArray):
        data.attrs = y.attrs
    return data


def is_empty(data):
    if isinstance(data, xr.DataArray):
        if data.shape == ():
            return True
        return False


def is_nodata(x):
    return x is None


def is_nan(x):
    if isinstance(x, (int, float)):
        return math.isnan(x)
    if isinstance(x, xr.DataArray):
        return x.isnull()


def is_valid(x):
    if x is None:
        return False
    elif isinstance(x, float):
        return not (math.isnan(x) or math.isinf(x))
    return True


def is_infinite(x):
    if x is None:
        return False
    return da.isinf(x)


def eq(
    x, y, delta=False, case_sensitive=True, reduce=False
):  # TODO: add equal checks for date strings in xar
    if x is None or y is None:
        return None

    x_type = (
        x.dtype if isinstance(x, (xr.core.dataarray.DataArray, np.ndarray)) else type(x)
    )
    y_type = (
        y.dtype if isinstance(y, (xr.core.dataarray.DataArray, np.ndarray)) else type(y)
    )

    if (x_type in [float, int]) and (
        y_type in [float, int]
    ):  # both arrays only contain numbers
        if type(delta) in [float, int]:
            ar_eq = abs(x - y) <= delta
        else:
            ar_eq = x == y
    else:
        ar_eq = x == y
    ar_eq = keep_attrs(x, y, ar_eq)
    if reduce:
        return ar_eq.all()
    else:
        return ar_eq


def neq(
    x, y, delta=None, case_sensitive=True, reduce=False
):  # TODO: add equal checks for date strings
    eq_val = eq(x, y, delta=delta, case_sensitive=case_sensitive, reduce=reduce)
    if eq_val is None:
        return None
    else:
        return da.logical_not(eq_val)


def gt(x, y, reduce=False):
    if x is None or y is None:
        return None
    gt_ar = x > y
    gt_ar = keep_attrs(x, y, gt_ar)
    if reduce:
        return gt_ar.all()
    else:
        return gt_ar


def gte(x, y, reduce=False):
    if x is None or y is None:
        return None
    gte_ar = (x - y) >= 0
    gte_ar = keep_attrs(x, y, gte_ar)
    if reduce:
        return gte_ar.all()
    else:
        return gte_ar


def lt(x, y, reduce=False):
    if x is None or y is None:
        return None
    lt_ar = x < y
    lt_ar = keep_attrs(x, y, lt_ar)
    if reduce:
        return lt_ar.all()
    else:
        return lt_ar


def lte(x, y, reduce=False):
    if x is None or y is None:
        return None
    lte_ar = x <= y
    lte_ar = keep_attrs(x, y, lte_ar)
    if reduce:
        return lte_ar.all()
    else:
        return lte_ar


def between(x, min, max, exclude_max=False, reduce=False):
    if x is None or min is None or max is None:
        return None
    if lt(max, min):
        return False
    if exclude_max:
        bet = da.logical_and(gte(x, min, reduce=reduce), lt(x, max, reduce=reduce))
    else:
        bet = da.logical_and(gte(x, min, reduce=reduce), lte(x, max, reduce=reduce))
    if isinstance(x, xr.DataArray):
        bet.attrs = x.attrs
    return bet
