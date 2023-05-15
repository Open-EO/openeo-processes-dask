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
            or type(x) in [np.ndarray, da.core.Array]
            and x.dtype.kind.lower() in ["u", "s"]
        ):
            return True
    if t == "list":
        if (
            isinstance(x, list)
            or type(x) in [np.ndarray, da.core.Array]
            and x.dtype.kind.lower() in ["u", "s"]
        ):
            return True
    if t == "dict":
        if (
            isinstance(x, dict)
            or type(x) in [np.ndarray, da.core.Array]
            and x.dtype.kind.lower() == "o"
        ):
            return True
    if t in ["int", "float"]:
        if (
            type(x) in [int, float]
            or type(x) in [np.ndarray, da.core.Array]
            and x.dtype.kind.lower() in ["i", "f"]
        ):
            return True
    return False


def is_array(x: ArrayLike, y: ArrayLike):
    if type(x) in [np.ndarray, da.core.Array] and type(y) in [
        np.ndarray,
        da.core.Array,
    ]:
        return True
    return False
