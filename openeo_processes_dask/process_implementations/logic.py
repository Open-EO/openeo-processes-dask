from typing import Callable, Optional, Union

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike

from openeo_processes_dask.process_implementations.comparison import is_nodata, is_valid

__all__ = ["and_", "or_", "xor", "not_", "if_", "any_", "all_"]


def and_(x: ArrayLike, y: ArrayLike):
    nan_x = is_nodata(x)
    nan_y = is_nodata(y)
    xy = np.logical_and(x, y)
    nan_mask = np.logical_and(nan_x, xy)
    xy = np.where(~nan_mask, xy, np.nan)
    nan_mask = np.logical_and(nan_y, xy)
    xy = np.where(~nan_mask, xy, np.nan)
    return xy


def or_(x: ArrayLike, y: ArrayLike):
    nan_x = is_nodata(x)
    nan_y = is_nodata(y)
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    xy = np.logical_or(x, y)
    nan_mask = np.logical_and(nan_x, np.logical_not(xy))
    xy = np.where(~nan_mask, xy, np.nan)
    nan_mask = np.logical_and(nan_y, np.logical_not(xy))
    xy = np.where(~nan_mask, xy, np.nan)
    return xy


def xor(x: ArrayLike, y: ArrayLike):
    nan_x = is_nodata(x)
    nan_y = is_nodata(y)
    xy = np.logical_xor(x, y)
    xy = np.where(~nan_x, xy, np.nan)
    xy = np.where(~nan_y, xy, np.nan)
    return xy


def not_(x: ArrayLike):
    not_x = np.logical_not(x)
    not_x = np.where(is_valid(x), not_x, np.nan)
    return not_x


def if_(
    value: ArrayLike,
    accept: Union[np.array, list, str, float, int],
    reject: Optional[Union[np.array, list, str, float, int]] = np.nan,
):
    return np.where(value, accept, reject)


def any_(
    data: ArrayLike,
    ignore_nodata: Optional[bool] = True,
    axis: Optional[int] = -1,
):
    if len(data) == 0:
        return np.nan
    data_any = np.any(np.nan_to_num(data), axis=axis)
    if not ignore_nodata:
        nan_ar = is_nodata(data)
        nan_mask = np.any(nan_ar, axis=axis)
        nan_mask = np.logical_and(nan_mask, ~data_any)
        data_any = np.where(~nan_mask, data_any, np.nan)
    return data_any


def all_(
    data: ArrayLike,
    ignore_nodata: Optional[bool] = True,
    axis: Optional[int] = -1,
):
    if len(data) == 0:
        return np.nan
    data_all = np.all(data, axis=axis)
    if not ignore_nodata:
        nan_ar = is_nodata(data)
        nan_mask = np.any(nan_ar, axis=axis)
        nan_mask = np.logical_and(nan_mask, data_all)
        data_all = np.where(~nan_mask, data_all, np.nan)
    return data_all
