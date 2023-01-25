from typing import Callable, Optional, Union

import dask.array as da
import numpy as np

__all__ = ["and_", "or_", "xor", "not_", "if_", "any_", "all_"]


def and_(x: Union[np.array, list], y: Union[np.array, list]):
    if not hasattr(x, "__array_interface__"):
        x = np.array(x)
    if not hasattr(y, "__array_interface__"):
        y = np.array(y)
    x = da.nan_to_num(x, copy=True, nan=False)
    y = da.nan_to_num(y, copy=True, nan=False)
    return da.logical_and(x, y)


def or_(x, y):
    if not hasattr(x, "__array_interface__"):
        x = np.array(x)
    if not hasattr(y, "__array_interface__"):
        y = np.array(y)
    x = da.nan_to_num(x, copy=True, nan=False)
    y = da.nan_to_num(y, copy=True, nan=False)
    return da.logical_or(x, y)


def xor(x, y):
    if not hasattr(x, "__array_interface__"):
        x = np.array(x)
    if not hasattr(y, "__array_interface__"):
        y = np.array(y)
    x = da.nan_to_num(x, copy=True, nan=False)
    y = da.nan_to_num(y, copy=True, nan=False)
    if x is None or y is None:
        return None
    return da.logical_xor(x, y)


def not_(x):
    if not hasattr(x, "__array_interface__"):
        x = np.array(x)
    not_x = da.logical_not(x)
    not_x = da.where(~da.isnan(x), not_x, np.nan)
    return not_x


def if_(value, accept, reject=np.nan):
    return da.where(value, accept, reject)


def any_(data, ignore_nodata=True, axis=-1):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    data_any = da.any(da.nan_to_num(data, nan=0), axis=axis)
    if not ignore_nodata:
        nan_ar = da.isnan(data)
        nan_mask = da.any(nan_ar, axis=axis)
        nan_mask = da.logical_and(nan_mask, ~data_any)
        data_any = da.where(~nan_mask, data_any, np.nan)
    return data_any


def all_(data, ignore_nodata=True, axis=None):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    data_all = da.all(data, axis=axis)
    if not ignore_nodata:
        nan_ar = da.isnan(data)
        nan_mask = da.any(nan_ar, axis=axis)
        nan_mask = da.logical_and(nan_mask, data_all)
        data_all = da.where(~nan_mask, data_all, np.nan)
    return data_all
