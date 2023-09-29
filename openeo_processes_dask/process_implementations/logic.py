from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from openeo_processes_dask.process_implementations.cubes.utils import isnull, notnull

__all__ = ["_and", "_or", "xor", "_not", "_if", "_any", "_all"]


def _and(x: ArrayLike, y: ArrayLike):
    nan_x = isnull(x)
    nan_y = isnull(y)
    xy = np.logical_and(x, y)
    nan_mask = np.logical_and(nan_x, xy)
    xy = np.where(~nan_mask, xy, np.nan)
    nan_mask = np.logical_and(nan_y, xy)
    xy = np.where(~nan_mask, xy, np.nan)
    return xy


def _or(x: ArrayLike, y: ArrayLike):
    nan_x = isnull(x)
    nan_y = isnull(y)
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    xy = np.logical_or(x, y)
    nan_mask = np.logical_and(nan_x, np.logical_not(xy))
    xy = np.where(~nan_mask, xy, np.nan)
    nan_mask = np.logical_and(nan_y, np.logical_not(xy))
    xy = np.where(~nan_mask, xy, np.nan)
    return xy


def xor(x: ArrayLike, y: ArrayLike):
    nan_x = isnull(x)
    nan_y = isnull(y)
    xy = np.logical_xor(x, y)
    xy = np.where(~nan_x, xy, np.nan)
    xy = np.where(~nan_y, xy, np.nan)
    return xy


def _not(x: ArrayLike):
    not_x = np.logical_not(x)
    not_x = np.where(notnull(x), not_x, np.nan)
    return not_x


def _if(
    value: ArrayLike,
    accept: Union[np.array, list, str, float, int],
    reject: Optional[Union[np.array, list, str, float, int]] = np.nan,
):
    return np.where(value, accept, reject)


def _any(
    data: ArrayLike,
    ignore_nodata: Optional[bool] = True,
    axis: Optional[int] = None,
):
    if len(data) == 0:
        return np.nan
    data_any = np.any(np.nan_to_num(data), axis=axis)
    if not ignore_nodata:
        nan_ar = isnull(data)
        nan_mask = np.any(nan_ar, axis=axis)
        nan_mask = np.logical_and(nan_mask, ~data_any)
        data_any = np.where(~nan_mask, data_any, np.nan)
    return data_any


def _all(
    data: ArrayLike,
    ignore_nodata: Optional[bool] = True,
    axis: Optional[int] = None,
):
    if len(data) == 0:
        return np.nan
    data_all = np.all(data, axis=axis)
    if not ignore_nodata:
        nan_ar = np.logical_not(notnull(data))
        nan_mask = np.any(nan_ar, axis=axis)
        nan_mask = np.logical_and(nan_mask, data_all)
        data_all = np.where(~nan_mask, data_all, np.nan)
    return data_all
