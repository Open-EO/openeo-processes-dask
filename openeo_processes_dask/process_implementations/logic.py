from typing import Callable, Optional, Union

import dask.array as da
import numpy as np

__all__ = ["and_", "or_", "xor", "not_", "if_", "any_", "all_"]


def and_(x: Union[np.array, list], y: Union[np.array, list]):
    if not hasattr(x, "__array_interface__"):
        x = np.array(x)
    if not hasattr(y, "__array_interface__"):
        y = np.array(y)
    x = np.nan_to_num(x, copy=True, nan=False)
    y = np.nan_to_num(y, copy=True, nan=False)
    return np.logical_and(x, y)


def or_(x, y):
    if not hasattr(x, "__array_interface__"):
        x = np.array(x)
    if not hasattr(y, "__array_interface__"):
        y = np.array(y)
    x = np.nan_to_num(x, copy=True, nan=False)
    y = np.nan_to_num(y, copy=True, nan=False)
    return np.logical_or(x, y)


def xor(x, y):
    if not hasattr(x, "__array_interface__"):
        x = np.array(x)
    if not hasattr(y, "__array_interface__"):
        y = np.array(y)
    x = np.nan_to_num(x, copy=True, nan=False)
    y = np.nan_to_num(y, copy=True, nan=False)
    if x is None or y is None:
        return None
    return np.logical_xor(x, y)


def not_(x):
    return da.logical_not(x)


def if_(value, accept, reject=np.nan):
    p = value.where(value == 0, accept)
    p = p.where(value == 1, reject)
    return p


def any_(data, ignore_nodata=True, dimension=None):
    data_nan = data.where(data == True, False)  # Set NaN to False
    if ignore_nodata:
        return data_nan.any(dim=dimension)
    else:
        data = data.any(dim=dimension)
        data_nan = data_nan.any(dim=dimension)
        if (data == data_nan).all():  # See if there are NaNs, that were set to False
            return data
        else:
            return data.where(data == data_nan, np.nan)


def all_(data, ignore_nodata=True, dimension=None):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    data_all = np.all(data, axis=dimension)
    if not ignore_nodata:
        nan_ar = np.isnan(data)
        nan_mask = np.any(nan_ar, axis=dimension)
        nan_mask = np.logical_and(nan_mask, data_all)
        data_all = np.where(~nan_mask, data_all, np.nan)
    return data_all
