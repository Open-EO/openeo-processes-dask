import logging
from typing import Callable, Optional, Union

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from openeo_processes_dask.exceptions import (
    ArrayElementNotAvailable,
    ArrayElementParameterConflict,
    ArrayElementParameterMissing,
)
from openeo_processes_dask.process_implementations.cubes.utils import _is_dask_array
from openeo_processes_dask.process_implementations.data_model import RasterCube

logger = logging.getLogger(__name__)


__all__ = [
    "array_element",
    "array_filter",
    "count",
    "array_create",
    "array_modify",
    "array_concat",
    "array_contains",
    "array_apply",
    "array_find",
    "array_labels",
    "first",
    "last",
    "order",
    "rearrange",
    "sort",
    "array_interpolate_linear",
]


def array_element(
    data: ArrayLike,
    index: Optional[int] = None,
    label: Optional[str] = None,
    return_nodata: Optional[bool] = False,
    axis=-1,
):
    if index is None and label is None:
        raise ArrayElementParameterMissing(
            "The process `array_element` requires either the `index` or `labels` parameter to be set."
        )

    if index is not None and label is not None:
        raise ArrayElementParameterConflict(
            "The process `array_element` only allows that either the `index` or the `labels` parameter is set."
        )

    if label is not None:
        raise NotImplementedError(
            "labelled arrays are currently not implemented. Please use index instead."
        )

    try:
        if index is not None:
            element = np.take(data, index, axis=axis)
            return element
    except IndexError:
        if return_nodata:
            logger.warning(
                f"Could not find index <{index}>, but return_nodata=True, so returning None."
            )
            output_shape = data.shape[0:axis] + data.shape[axis + 1 :]
            if _is_dask_array(data):
                result = da.empty(output_shape)
            else:
                result = np.empty(output_shape)
            result[:] = np.nan
            return result
        else:
            raise ArrayElementNotAvailable(
                f"The array has no element with the specified index or label: {index if index is not None else label}"
            )

    raise ValueError("Shouldn't have come here!")


def array_filter(data: RasterCube, condition: Callable, **kwargs):
    mask = condition(x=data, **kwargs)
    data = data[mask]
    return data


def count(data: RasterCube, condition: Callable, **kwargs):
    data = condition(x=data, **kwargs)
    if "dimension" in kwargs:
        if kwargs["dimension"] == "t":
            kwargs["dimension"] = "time"
        data = data.sum(dim=kwargs["dimension"])
    else:
        data = data.sum()
    return data


def array_create(data: ArrayLike, repeat: Optional[int] = 1, **kwargs):
    if type(data) in [int, float]:
        data = [data]
    if len(data) == 0:
        return np.array([])
    return np.tile(data, reps=repeat)


def array_modify(
    data: ArrayLike,
    values: ArrayLike,
    index: int,
    length: Optional[int] = 1,
    **kwargs,
):
    if index == 0:
        modified = values
    else:
        first = data[:index]
        modified = np.append(first, values)
    if index + length < len(data):
        modified = np.append(modified, data[index + length :])
    return modified


def array_concat(array1: ArrayLike, array2: ArrayLike, **kwargs):
    concat = np.append(array1, array2)
    return concat


def array_contains(data: ArrayLike, value: Union[int, float, str, list]):
    if np.array(pd.isnull(value)).all():
        return np.isnan(data).any()
    else:
        return np.isin(data, value).any()


def array_apply(
    data: ArrayLike,
    process: Callable,
    context: Optional[dict] = None,
    **kwargs,
):
    context = context if context is not None else {}
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    return process(data, **context)


def array_find(
    data: ArrayLike,
    value: float,
    reverse: Optional[bool] = False,
    axis: Optional[int] = None,
):
    if np.isnan(value) or len(data) == 0:
        return np.nan
    else:
        idxs = np.argmax((data == value), axis=axis)
    if reverse:
        b = np.flip(data, axis=axis)
        idxs = np.shape(b)[axis] - np.argmax((b == value), axis=axis) - 1
    return idxs


def array_labels(data: ArrayLike, dimension: Optional[int] = None):
    if dimension is None:
        n_vals = len(data)
    if isinstance(dimension, int):
        n_vals = np.shape(data)[dimension]
    return np.arange(n_vals)


def first(
    data: ArrayLike,
    ignore_nodata: Optional[bool] = True,
    axis: Optional[str] = None,
):
    if len(data) == 0:
        return np.nan
    if axis is None:
        axis = 0
    if ignore_nodata:  # skip np.nan values
        nan_mask = ~np.isnan(data)  # create mask for valid values (not np.nan)
        idx_first = np.argmax(nan_mask, axis=axis)
        first_elem = np.take(data, indices=0, axis=axis)
        if np.isnan(first_elem).any():
            for i in range(np.max(idx_first) + 1):
                first_elem = np.nan_to_num(first_elem, True, np.take(data, i, axis))
    else:  # take the first element, no matter np.nan values are in the array
        first_elem = np.take(data, indices=0, axis=axis)
    return first_elem


def last(
    data: ArrayLike,
    ignore_nodata: Optional[bool] = True,
    axis: Optional[str] = None,
):
    if len(data) == 0:
        return np.nan
    if axis is None:
        axis = 0
    data = np.flip(data, axis=axis)  # flip data to retrieve the first valid element
    last_elem = first(data, ignore_nodata=ignore_nodata, axis=axis)
    return last_elem


def order(
    data: ArrayLike,
    asc: Optional[bool] = True,
    nodata: Optional[bool] = True,
    axis: Optional[int] = None,
):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    if axis is None:
        axis = 0
    if asc:
        permutation_idxs = np.argsort(data, kind="mergesort", axis=axis)
    else:  # [::-1] not possible
        permutation_idxs = np.argsort(
            -data, kind="mergesort", axis=axis
        )  # to get the indizes in descending order, the sign of the data is changed

    if nodata is None:  # ignore np.nan values
        # sort the original data first, to get correct position of no data values
        sorted_data = data[permutation_idxs]
        return permutation_idxs[~pd.isnull(sorted_data)]
    elif nodata is False:  # put location/index of np.nan values first
        # sort the original data first, to get correct position of no data values
        sorted_data = data[permutation_idxs]
        nan_idxs = pd.isnull(sorted_data)

        # flip permutation and nan mask
        permutation_idxs_flip = np.flip(permutation_idxs, axis=axis)
        nan_idxs_flip = np.flip(nan_idxs, axis=axis)

        # flip causes the nan.values to be first, however the order of all other values is also flipped
        # therefore the non np.nan values (i.e. the wrong flipped order) is replaced by the right order given by
        # the original permutation values
        permutation_idxs_flip[~nan_idxs_flip] = permutation_idxs[~nan_idxs]

        return permutation_idxs_flip
    elif nodata is True:  # default argsort behaviour, np.nan values are put last
        return permutation_idxs
    else:
        err_msg = "Data type of 'nodata' argument is not supported."
        raise Exception(err_msg)


def rearrange(data: ArrayLike, order):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    return data[order]


def sort(
    data: ArrayLike,
    asc: Optional[bool] = True,
    nodata: Optional[bool] = None,
    axis: Optional[int] = None,
):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    if asc:
        data_sorted = np.sort(data, axis=axis)
    else:  # [::-1] not possible
        data_sorted = -np.sort(
            -data, axis=axis
        )  # to get the indexes in descending order, the sign of the data is changed

    if nodata is None:  # ignore np.nan values
        nan_idxs = pd.isnull(data_sorted)
        return data_sorted[~nan_idxs]
    elif nodata == False:  # put np.nan values first
        nan_idxs = pd.isnull(data_sorted)
        data_sorted_flip = np.flip(data_sorted, axis=axis)
        nan_idxs_flip = pd.isnull(data_sorted_flip)
        data_sorted_flip[~nan_idxs_flip] = data_sorted[~nan_idxs]
        return data_sorted_flip
    elif nodata == True:  # default sort behaviour, np.nan values are put last
        return data_sorted
    else:
        err_msg = "Data type of 'nodata' argument is not supported."
        raise Exception(err_msg)


def array_interpolate_linear(data: ArrayLike, axis: Optional[int] = -1):
    data_move = np.moveaxis(data, axis, -1)
    data_flat = np.reshape(data_move, -1)
    xp = np.arange(len(data_flat))
    interp_flat = np.interp(
        x=xp, xp=xp[~np.isnan(data_flat)], fp=data_flat[~np.isnan(data_flat)]
    )
    interpolated = np.moveaxis(np.reshape(interp_flat, np.shape(data_move)), -1, 0)
    if np.isnan(np.take(data, indices=0, axis=axis)).any():
        interpolated[0, :] = np.take(data, indices=0, axis=axis)
    if np.isnan(np.take(data, indices=-1, axis=axis)).any():
        interpolated[-1, :] = np.take(data, indices=-1, axis=axis)
    return np.moveaxis(interpolated, 0, axis)
