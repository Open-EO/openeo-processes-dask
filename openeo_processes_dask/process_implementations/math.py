import dask
import dask.array as da
import numpy as np

from openeo_processes_dask.process_implementations.cubes.utils import (
    _has_dask,
    _is_dask_array,
)
from openeo_processes_dask.process_implementations.exceptions import (
    OpenEOException,
    QuantilesParameterConflict,
    QuantilesParameterMissing,
)

__all__ = [
    "e",
    "pi",
    "nan",
    "constant",
    "divide",
    "subtract",
    "multiply",
    "add",
    "_sum",
    "_min",
    "_max",
    "median",
    "mean",
    "sd",
    "variance",
    "floor",
    "ceil",
    "_int",
    "_round",
    "exp",
    "log",
    "ln",
    "cos",
    "arccos",
    "cosh",
    "arcosh",
    "sin",
    "arcsin",
    "sinh",
    "arsinh",
    "tan",
    "arctan",
    "tanh",
    "artanh",
    "arctan2",
    "linear_scale_range",
    "mod",
    "absolute",
    "sgn",
    "sqrt",
    "power",
    "extrema",
    "clip",
    "quantiles",
    "product",
    "normalized_difference",
]


def e():
    return np.e


def pi():
    return np.pi


def nan(data=None):
    if data is None or isinstance(data, np.ndarray):
        return np.nan
    elif data is not None and _has_dask() and _is_dask_array(data):
        return dask.delayed(np.nan)
    else:
        raise OpenEOException(
            f"Don't know which nan-value to use for provided datatype: {type(data)}."
        )


def constant(x):
    return x


def divide(x, y):
    result = x / y
    return result


def subtract(x, y):
    result = x - y
    return result


def multiply(x, y):
    result = x * y
    return result


def add(x, y):
    result = x + y
    return result


def _min(data, ignore_nodata=True, axis=None, keepdims=False):
    if ignore_nodata:
        return np.nanmin(data, axis=axis, keepdims=keepdims)
    else:
        return np.min(data, axis=axis, keepdims=keepdims)


def _max(data, ignore_nodata=True, axis=None, keepdims=False):
    if ignore_nodata:
        return np.nanmax(data, axis=axis, keepdims=keepdims)
    else:
        return np.max(data, axis=axis, keepdims=keepdims)


def median(data, ignore_nodata=True, axis=None, keepdims=False):
    if ignore_nodata:
        return np.nanmedian(data, axis=axis, keepdims=keepdims)
    else:
        return np.median(data, axis=axis, keepdims=keepdims)


def mean(data, ignore_nodata=True, axis=None, keepdims=False):
    if ignore_nodata:
        return np.nanmean(data, axis=axis, keepdims=keepdims)
    else:
        return np.mean(data, axis=axis, keepdims=keepdims)


def sd(data, ignore_nodata=True, axis=None, keepdims=False):
    if ignore_nodata:
        return np.nanstd(data, axis=axis, ddof=1, keepdims=keepdims)
    else:
        return np.std(data, axis=axis, ddof=1, keepdims=keepdims)


def variance(data, ignore_nodata=True, axis=None, keepdims=False):
    if ignore_nodata:
        return np.nanvar(data, axis=axis, ddof=1, keepdims=keepdims)
    else:
        return np.var(data, axis=axis, ddof=1, keepdims=keepdims)


def floor(x):
    return np.floor(x)


def ceil(x):
    return np.ceil(x)


def _int(x):
    return np.trunc(x)


def _round(x, p=0):
    return np.around(x, decimals=p)


def exp(p):
    return np.exp(p)


def log(x, base):
    return np.log(x) / np.log(base)


def ln(x):
    return np.log(x)


def cos(x):
    return np.cos(x)


def arccos(x):
    return np.arccos(x)


def cosh(x):
    return np.cosh(x)


def arcosh(x):
    return np.arccosh(x)


def sin(x):
    return np.sin(x)


def arcsin(x):
    return np.arcsin(x)


def sinh(x):
    return np.sinh(x)


def arsinh(x):
    return np.arcsinh(x)


def tan(x):
    return np.tan(x)


def arctan(x):
    return np.arctan(x)


def tanh(x):
    return np.tanh(x)


def artanh(x):
    return np.arctanh(x)


def arctan2(y, x):
    return np.arctan2(y, x)


def linear_scale_range(x, inputMin, inputMax, outputMin=0.0, outputMax=1.0):
    x = clip(x, inputMin, inputMax)
    lsr = ((x - inputMin) / (inputMax - inputMin)) * (outputMax - outputMin) + outputMin
    return lsr


def mod(x, y):
    return np.mod(x, y)


def absolute(x):
    return np.abs(x)


def sgn(x):
    return np.sign(x)


def sqrt(x):
    return np.sqrt(x)


def power(base, p):
    e = base**p
    return e


def extrema(data, ignore_nodata=True, axis=None, keepdims=False):
    # TODO: Could be sped up by only iterating over array once
    minimum = _min(data, skipna=ignore_nodata, axis=axis, keepdims=keepdims)
    maximum = _max(data, skipna=ignore_nodata, axis=axis, keepdims=keepdims)
    array = dask.delayed(np.array)([minimum, maximum])
    return da.from_delayed(array, (2,), dtype=data.dtype)


def clip(x, min, max):
    # Cannot use positional arguments to pass min and max into np.clip, this will not work with dask.
    return np.clip(x, min, max)


def quantiles(
    data, probabilities=None, q=None, ignore_nodata=True, axis=None, keepdims=False
):
    if probabilities is not None and q is not None:
        raise QuantilesParameterConflict(
            "The process `quantiles` requires either the `probabilities` or `q` parameter to be set."
        )

    if probabilities is None and q is None:
        raise QuantilesParameterMissing(
            "The process `quantiles` only allows that either the `probabilities` or the `q` parameter is set."
        )

    if isinstance(probabilities, list):
        probabilities = np.array(probabilities)

    if q is not None:
        probabilities = np.arange(1.0 / q, 1, 1.0 / q)

    if data.size == 0:
        return np.array([np.nan] * len(probabilities))

    if ignore_nodata:
        result = np.nanquantile(
            data, q=probabilities, method="linear", axis=axis, keepdims=keepdims
        )
    else:
        result = np.quantile(
            data, q=probabilities, method="linear", axis=axis, keepdims=keepdims
        )

    return result


def _sum(data, ignore_nodata=True, axis=None, keepdims=False):
    if len(data) == 0:
        return nan(data=data)

    if ignore_nodata:
        result = np.nansum(data, axis=axis, keepdims=keepdims)
    else:
        result = np.sum(data, axis=axis, keepdims=keepdims)
    return result


def product(data, ignore_nodata=True, axis=None, keepdims=False):
    if len(data) == 0:
        return nan(data=data)

    if ignore_nodata:
        result = np.nanprod(data, axis=axis, keepdims=keepdims)
    else:
        result = np.prod(data, axis=axis, keepdims=keepdims)
    return result


def normalized_difference(x, y):
    nd = (x - y) / (x + y)
    return nd
