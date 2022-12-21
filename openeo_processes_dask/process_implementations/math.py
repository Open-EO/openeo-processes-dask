import numbers

import dask
import dask.array as da
import numpy as np
import xarray as xr

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
    "int",
    "round",
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
    "scale",
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
    "ndvi",
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


def e():
    return np.e


def pi():
    return np.pi


def nan():
    return np.nan


def constant(x):
    return x


def divide(x, y, **kwargs):
    result = x / y
    return result


def subtract(x, y, **kwargs):
    result = x - y
    return result


def multiply(x, y, **kwargs):
    result = x * y
    return result


def add(x, y, **kwargs):
    result = x + y
    return result


def _min(data, ignore_nodata=True, axis=-1, **kwargs):
    if ignore_nodata:
        return np.nanmin(data, axis=axis)
    else:
        return np.min(data, axis=axis)


def _max(data, ignore_nodata=True, axis=-1, **kwargs):
    if ignore_nodata:
        return np.nanmax(data, axis=axis)
    else:
        return np.max(data, axis=axis)


def median(data, ignore_nodata=True, axis=-1, **kwargs):
    if ignore_nodata:
        return np.nanmedian(data, axis=axis)
    else:
        return np.median(data, axis=axis)


def mean(data, ignore_nodata=False, axis=-1, **kwargs):
    if ignore_nodata:
        return np.nanmean(data, axis=axis)
    else:
        return np.mean(data, axis=axis)


def sd(data, ignore_nodata=False, axis=-1, **kwargs):
    if ignore_nodata:
        return np.nanstd(data, axis=axis, ddof=1)
    else:
        return np.std(data, axis=axis, ddof=1)


def variance(data, ignore_nodata=False, axis=-1, **kwargs):
    if ignore_nodata:
        return np.nanvar(data, axis=axis, ddof=1)
    else:
        return np.var(data, axis=axis, ddof=1)


def floor(x):
    return da.floor(x)


def ceil(x):
    return da.ceil(x)


def int(x):
    return da.trunc(x)


def round(x, p=0):
    return x.round(p)


def exp(p):
    return da.exp(p)


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
    lsr = ((x - inputMin) / (inputMax - inputMin)) * (outputMax - outputMin) + outputMin
    return lsr


def scale(x, factor=1.0):
    s = x * factor
    return s


def mod(x, y):
    if x is None or y is None:
        return np.nan
    m = x % y
    return m


def absolute(x):
    return np.abs(x)


def sgn(x):
    return np.sign(x)


def sqrt(x):
    return np.sqrt(x)


def power(base, p):
    e = base**p
    return e


def extrema(data, ignore_nodata=True, axis=-1):
    if isinstance(data, xr.DataArray):
        data = data.data

    # TODO: Could be sped up by only iterating over array once
    minimum = _min(data, skipna=ignore_nodata, axis=axis)
    maximum = _max(data, skipna=ignore_nodata, axis=axis)
    array = dask.delayed(np.array)([minimum, maximum])
    return da.from_delayed(array, (2,), dtype=data.dtype)


def clip(x, min, max):
    return np.clip(x, a_min=min, a_max=max)


def quantiles(data, probabilities=None, q=None, ignore_nodata=True, dimension=None):
    if probabilities is not None and q is not None:
        raise Exception(
            "QuantilesParameterConflict: The process `quantiles` only allows that either the `probabilities` or the `q` parameter is set."
        )

    if q is not None:
        probabilities = list(np.arange(0, 1, 1.0 / q))[1:]
    q = data.quantile(np.array(probabilities), dim=dimension, skipna=ignore_nodata)
    q.attrs = data.attrs
    return q


def _sum(data, ignore_nodata=True, dimension=None):
    summand = 0
    if isinstance(data, list):
        data_tmp = []
        for item in data:
            if isinstance(item, xr.DataArray):
                data_tmp.append(item)
            elif isinstance(item, numbers.Number):
                summand += item
        # Concatenate along dim 'new_dim'
        data = xr.concat(data_tmp, dim="new_dim")
        return data.sum(dim="new_dim", skipna=ignore_nodata) + summand

    if isinstance(data, xr.DataArray):
        if not dimension:
            dimension = data.dims[0]
        s = data.sum(dim=dimension, skipna=ignore_nodata)
        s.attrs = data.attrs
        return s


def product(data, ignore_nodata=True, dimension=None, extra_values=None):
    extra_values = extra_values if extra_values is not None else []
    if len(extra_values) > 0:
        multiplicand = np.prod(extra_values)
    else:
        multiplicand = 1.0
    p = data.prod(dim=dimension, skipna=ignore_nodata) * multiplicand
    p.attrs = data.attrs
    return p


def normalized_difference(x, y):
    nd = (x - y) / (x + y)
    return nd


def ndvi(data, nir="nir", red="red", target_band=None):
    r = np.nan
    n = np.nan
    if "bands" in data.dims:
        if red == "red":
            if "B04" in data["bands"].values:
                r = data.sel(bands="B04")
        elif red == "rededge":
            if "B05" in data["bands"].values:
                r = data.sel(bands="B05")
            elif "B06" in data["bands"].values:
                r = data.sel(bands="B06")
            elif "B07" in data["bands"].values:
                r = data.sel(bands="B07")
        if nir == "nir":
            n = data.sel(bands="B08")
        elif nir == "nir08":
            if "B8a" in data["bands"].values:
                n = data.sel(bands="B8a")
            elif "B8A" in data["bands"].values:
                n = data.sel(bands="B8A")
            elif "B05" in data["bands"].values:
                n = data.sel(bands="B05")
        elif nir == "nir09":
            if "B09" in data["bands"].values:
                n = data.sel(bands="B09")
        if red in data["bands"].values:
            r = data.sel(bands=red)
        if nir in data["bands"].values:
            n = data.sel(bands=nir)
    nd = normalized_difference(n, r)
    if target_band is not None:
        nd = nd.assign_coords(bands=target_band)
    # TODO: Remove this once we have the .openeo accessor
    nd.attrs = data.attrs
    return nd
