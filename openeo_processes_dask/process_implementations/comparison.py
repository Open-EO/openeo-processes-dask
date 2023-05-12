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
