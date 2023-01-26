from typing import Callable, Optional, Union

import dask.array as da
import numpy as np

__all__ = ["and_", "or_", "xor", "not_", "if_", "any_", "all_"]


def and_(x: Union[np.array, list], y: Union[np.array, list]):
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    return np.logical_and(x, y)


def or_(x: Union[np.array, list], y: Union[np.array, list]):
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    return np.logical_or(x, y)


def xor(x: Union[np.array, list], y: Union[np.array, list]):
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    if x is None or y is None:
        return None
    return np.logical_xor(x, y)


def not_(x: Union[np.array, list]):
    not_x = np.logical_not(x)
    not_x = np.where(~np.isnan(x), not_x, np.nan)
    return not_x


def if_(
    value: Union[np.array, list],
    accept: Union[np.array, list, str, float, int],
    reject: Optional[Union[np.array, list, str, float, int]] = np.nan,
):
    return np.where(value, accept, reject)


def any_(
    data: Union[np.array, list],
    ignore_nodata: Optional[bool] = True,
    axis: Optional[int] = -1,
):
    if len(data) == 0:
        return np.nan
    data_any = np.any(np.nan_to_num(data), axis=axis)
    if not ignore_nodata:
        nan_ar = np.isnan(data)
        nan_mask = np.any(nan_ar, axis=axis)
        nan_mask = np.logical_and(nan_mask, ~data_any)
        data_any = np.where(~nan_mask, data_any, np.nan)
    return data_any


def all_(
    data: Union[np.array, list],
    ignore_nodata: Optional[bool] = True,
    axis: Optional[int] = -1,
):
    if len(data) == 0:
        return np.nan
    data_all = np.all(data, axis=axis)
    if not ignore_nodata:
        nan_ar = np.isnan(data)
        nan_mask = np.any(nan_ar, axis=axis)
        nan_mask = np.logical_and(nan_mask, data_all)
        data_all = np.where(~nan_mask, data_all, np.nan)
    return data_all
