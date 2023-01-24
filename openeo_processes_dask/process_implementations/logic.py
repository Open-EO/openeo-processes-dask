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
    if not hasattr(x, "__array_interface__"):
        x = np.array(x)
    not_x = np.logical_not(x)
    not_x = np.where(~np.isnan(x), not_x, np.nan)
    return not_x


def if_(value, accept, reject=np.nan):
    return np.where(value, accept, reject)


def any_(data, ignore_nodata=True, dimension=None):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    data_any = np.any(np.nan_to_num(data, nan=0), axis=dimension)
    if not ignore_nodata:
        nan_ar = np.isnan(data)
        nan_mask = np.any(nan_ar, axis=dimension)
        nan_mask = np.logical_and(nan_mask, ~data_any)
        data_any = np.where(~nan_mask, data_any, np.nan)
    return data_any


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
