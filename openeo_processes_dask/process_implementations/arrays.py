from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from openeo_processes_dask.exceptions import (
    ArrayElementParameterConflict,
    ArrayElementParameterMissing,
)
from openeo_processes_dask.process_implementations.data_model import RasterCube

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
]


def array_element(
    data: Union[xr.Dataset, xr.DataArray, list],
    index: Optional[int] = None,
    label: Optional[str] = None,
    return_nodata: Optional[bool] = False,
    dimension: Optional[str] = None,
    **kwargs
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
        element = data.sel({dimension: label})
        return element

    if index is not None:
        if dimension is not None:
            element = data.isel({dimension: int(index)})
            return element
        else:
            element = data[int(index)]
            return element

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


def array_create(
    data: Union[np.array, list, float, int], repeat: Optional[int] = 1, **kwargs
):
    if type(data) in [int, float]:
        data = [data]
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.array([])
    return np.tile(data, reps=repeat)


def array_modify(
    data: Union[np.array, list],
    values: Union[np.array, list],
    index: int,
    length: Optional[int] = 1,
    **kwargs
):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if not hasattr(values, "__array_interface__"):
        values = np.array(values)
    if index == 0:
        modified = values
    else:
        first = data[:index]
        modified = np.append(first, values)
    if index + length < len(data):
        modified = np.append(modified, data[index + length :])
    return modified


def array_concat(
    array1: Union[np.array, list], array2: Union[np.array, list], **kwargs
):
    if not hasattr(array1, "__array_interface__"):
        array1 = np.array(array1)
    if not hasattr(array2, "__array_interface__"):
        array2 = np.array(array2)
    concat = np.append(array1, array2)
    return concat


def array_contains(data: Union[np.array, list], value: Union[int, float, str, list]):
    if np.array(pd.isnull(value)).all():
        return np.isnan(data).any()
    else:
        return np.isin(data, value).any()


def array_apply(
    data: Union[np.array, list],
    process: Callable,
    context: Optional[dict] = None,
    **kwargs
):
    context = context if context is not None else {}
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    return process(data, **context)


def array_find(
    data: Union[np.array, list],
    value: float,
    reverse: Optional[bool] = False,
    dimension: Optional[int] = None,
):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if np.isnan(value) or len(data) == 0:
        return np.nan
    else:
        idxs = np.argmax((data == value), axis=dimension)
    if reverse:
        b = np.flip(data, axis=dimension)
        idxs = np.shape(b)[dimension] - np.argmax((b == value), axis=dimension) - 1
    return idxs


def array_labels(data: Union[np.array, list], dimension: Optional[int] = None):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if dimension is None:
        n_vals = len(data)
    if isinstance(dimension, int):
        n_vals = np.shape(data)[dimension]
    return np.arange(n_vals)


def first(
    data: Union[np.array, list],
    ignore_nodata: Optional[bool] = True,
    dimension: Optional[str] = None,
):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    if dimension is None:
        dimension = 0
    n_dims = len(data.shape)
    if ignore_nodata:  # skip np.nan values
        nan_mask = ~pd.isnull(data)  # create mask for valid values (not np.nan)
        idx_first = np.argmax(nan_mask, axis=dimension)
        first_elem = np.take_along_axis(
            data, np.expand_dims(idx_first, axis=dimension), axis=dimension
        )
    else:  # take the first element, no matter np.nan values are in the array
        slices = [slice(None)] * n_dims
        slices[dimension] = 0
        idx_first = tuple(slices)
        first_elem = data[idx_first]

    return first_elem


def last(
    data: Union[np.array, list],
    ignore_nodata: Optional[bool] = True,
    dimension: Optional[str] = None,
):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    if dimension is None:
        dimension = 0
    n_dims = len(data.shape)
    if ignore_nodata:  # skip np.nan values
        data = np.flip(
            data, axis=dimension
        )  # flip data to retrieve the first valid element (thats the only way it works with argmax)
        last_elem = first(data, ignore_nodata=ignore_nodata, dimension=dimension)
    else:  # take the first element, no matter np.nan values are in the array
        slices = [slice(None)] * n_dims
        slices[dimension] = -1
        idx_last = tuple(slices)
        last_elem = data[idx_last]

    return last_elem


def order(
    data: Union[np.array, list],
    asc: Optional[bool] = True,
    nodata: Optional[bool] = True,
    dimension: Optional[int] = None,
):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    if dimension is None:
        dimension = 0
    if asc:
        permutation_idxs = np.argsort(data, kind="mergesort", axis=dimension)
    else:  # [::-1] not possible
        permutation_idxs = np.argsort(
            -data, kind="mergesort", axis=dimension
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
        permutation_idxs_flip = np.flip(permutation_idxs, axis=dimension)
        nan_idxs_flip = np.flip(nan_idxs, axis=dimension)

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


def rearrange(data: Union[np.array, list], order):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    return data[order]


def sort(
    data: Union[np.array, list],
    asc: Optional[bool] = True,
    nodata: Optional[bool] = None,
    dimension: Optional[int] = None,
):
    if not hasattr(data, "__array_interface__"):
        data = np.array(data)
    if len(data) == 0:
        return np.nan
    if asc:
        data_sorted = np.sort(data, axis=dimension)
    else:  # [::-1] not possible
        data_sorted = -np.sort(
            -data, axis=dimension
        )  # to get the indexes in descending order, the sign of the data is changed

    if nodata is None:  # ignore np.nan values
        nan_idxs = pd.isnull(data_sorted)
        return data_sorted[~nan_idxs]
    elif nodata == False:  # put np.nan values first
        nan_idxs = pd.isnull(data_sorted)
        data_sorted_flip = np.flip(data_sorted, axis=dimension)
        nan_idxs_flip = pd.isnull(data_sorted_flip)
        data_sorted_flip[~nan_idxs_flip] = data_sorted[~nan_idxs]
        return data_sorted_flip
    elif nodata == True:  # default sort behaviour, np.nan values are put last
        return data_sorted
    else:
        err_msg = "Data type of 'nodata' argument is not supported."
        raise Exception(err_msg)
