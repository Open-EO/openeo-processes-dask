import datetime
from typing import Callable, Optional

import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from xarray.core.duck_array_ops import notnull


def check_type(x, t="float"):
    if t == "str":
        if (
            isinstance(x, str)
            or type(x) in [np.ndarray, xr.DataArray, da.core.Array]
            and x.dtype.kind.lower() in ["u", "s"]
        ):
            return True
    if t == "list":
        if (
            isinstance(x, list)
            or type(x) in [np.ndarray, xr.DataArray, da.core.Array]
            and x.dtype.kind.lower() in ["u", "s"]
        ):
            return True
    if t == "dict":
        if (
            isinstance(x, dict)
            or type(x) in [np.ndarray, xr.DataArray, da.core.Array]
            and x.dtype.kind.lower() == "o"
        ):
            return True
    if t in ["int", "float"]:
        if (
            type(x) in [int, float]
            or type(x) in [np.ndarray, xr.DataArray, da.core.Array]
            and x.dtype.kind.lower() in ["i", "f"]
        ):
            return True
    if t == "bool":
        if (
            isinstance(x, bool)
            or type(x) in [np.ndarray, xr.DataArray, da.core.Array]
            and x.dtype.kind.lower() == "b"
        ):
            return True
    return False


def is_number(x: ArrayLike, y: ArrayLike = 0):
    # check if x and y are only one value each
    return (
        type(x) in [int, float, bool]
        and type(y) in [int, float, bool]
        or isinstance(x, list)
        and len(x) == 1
        and isinstance(y, list)
        and len(y) == 1
    )
