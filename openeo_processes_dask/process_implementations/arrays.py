import itertools
import logging
from typing import Any, Optional

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from xarray.core.duck_array_ops import isnull, notnull

from openeo_processes_dask.process_implementations.cubes.utils import _is_dask_array
from openeo_processes_dask.process_implementations.exceptions import (
    ArrayElementNotAvailable,
    ArrayElementParameterConflict,
    ArrayElementParameterMissing,
    TooManyDimensions,
)

logger = logging.getLogger(__name__)


__all__ = [
    "array_element",
    "array_create",
    "array_modify",
    "array_concat",
    "array_contains",
    "array_find",
    "array_labels",
    "first",
    "last",
    "order",
    "rearrange",
    "sort",
]


def array_element(
    data: ArrayLike,
    index: Optional[int] = None,
    label: Optional[str] = None,
    return_nodata: Optional[bool] = False,
    axis=None,
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


def array_create(
    data: Optional[ArrayLike] = None, repeat: Optional[int] = 1
) -> ArrayLike:
    if data is None:
        data = np.array([])
    return np.tile(data, reps=repeat)


def array_modify(
    data: ArrayLike,
    values: ArrayLike,
    index: int,
    length: Optional[int] = 1,
) -> ArrayLike:
    if index > len(data):
        raise ArrayElementNotAvailable(
            "The array can't be modified as the given index is larger than the number of elements in the array."
        )
    first = data[:index]
    modified = np.append(first, values)
    if index + length < len(data):
        modified = np.append(modified, data[index + length :])
    return modified


def array_concat(array1: ArrayLike, array2: ArrayLike) -> ArrayLike:
    if isinstance(array1, list):
        array1 = np.asarray(array1)
    if isinstance(array2, list):
        array2 = np.asarray(array2)

    concat = np.concatenate([array1, array2])

    # e.g. concating int32 and str arrays results in the result being cast to a Unicode dtype of a certain length (e.g. <U22).
    # There isn't really anything better to do as numpy does not support heterogenuous arrays.
    # Best we can do at this point is to at least make the user aware that this is what has happened.
    if array1.dtype.kind != array2.dtype.kind:
        logger.warning(
            f"array_concat: different datatypes for array1 ({array1.dtype}) and array2 ({array2.dtype}), cast to {concat.dtype}"
        )

    return concat


def array_contains(data: ArrayLike, value: Any, axis=None) -> bool:
    # TODO: Contrary to the process spec, our implementation does interpret temporal strings before checking them here
    # This is somewhat implicit in how we currently parse parameters, so cannot be easily changed.

    value_is_valid = False
    valid_dtypes = [np.number, np.bool_, np.str_]
    for dtype in valid_dtypes:
        if np.issubdtype(type(value), dtype):
            value_is_valid = True
    if len(np.shape(data)) != 1 and axis is None:
        return False
    if not value_is_valid:
        return False
    if pd.isnull(value):
        return np.isnan(data).any(axis=axis)
    else:
        return np.isin(data, value).any(axis=axis)


def array_find(
    data: ArrayLike,
    value: Any,
    reverse: Optional[bool] = False,
    axis: Optional[int] = None,
) -> np.number:
    if isinstance(data, list):
        data = np.asarray(data)

    if reverse:
        data = np.flip(data, axis=axis)

    idxs = (data == value).argmax(axis=axis)

    mask = ~np.array((data == value).any(axis=axis))
    if np.isnan(value):
        mask = True

    logger.warning(
        "array_find: numpy has no sentinel value for missing data in integer arrays, therefore np.masked_array is used to return the indices of found elements. Further operations might fail if not defined for masked arrays."
    )
    if isinstance(idxs, da.Array):
        idxs = idxs.compute_chunk_sizes()
        masked_idxs = np.atleast_1d(da.ma.masked_array(idxs, mask=mask))
    else:
        masked_idxs = np.atleast_1d(np.ma.masked_array(idxs, mask=mask))

    return masked_idxs


def array_labels(data: ArrayLike) -> ArrayLike:
    logger.warning(
        "Labelled arrays are currently not supported, array_labels will only return indices."
    )
    if isinstance(data, list):
        data = np.asarray(data)
    if len(data.shape) > 1:
        raise TooManyDimensions("array_labels is only implemented for 1D arrays.")
    return np.arange(len(data))


def first(
    data: ArrayLike,
    ignore_nodata: Optional[bool] = True,
    axis: Optional[str] = None,
):
    if len(data) == 0:
        return np.nan
    if axis is None:
        data = data.flatten()
        axis = 0
    if ignore_nodata:
        nan_mask = ~pd.isnull(data)  # create mask for valid values (not np.nan)
        idx_first = np.argmax(nan_mask, axis=axis)
        first_elem = np.take(data, indices=0, axis=axis)

        if pd.isnull(np.asarray(first_elem)).any():
            for i in range(np.max(idx_first) + 1):
                first_elem = np.nan_to_num(first_elem, True, np.take(data, i, axis))
    else:  # take the first element, no matter np.nan values are in the array
        first_elem = np.take(data, indices=0, axis=axis)
    return first_elem


def last(
    data: ArrayLike,
    ignore_nodata: Optional[bool] = True,
    axis: Optional[int] = None,
):
    if len(data) == 0:
        return np.nan
    data = np.flip(data, axis=axis)  # flip data to retrieve the first valid element
    last_elem = first(data, ignore_nodata=ignore_nodata, axis=axis)
    return last_elem


def order(
    data: ArrayLike,
    asc: Optional[bool] = True,
    nodata: Optional[bool] = None,
    axis: Optional[int] = None,
):
    if isinstance(data, list):
        data = np.asarray(data)
    if len(data) == 0:
        return data

    # See https://github.com/dask/dask/issues/4368
    logger.warning(
        "order: Dask does not support lazy sorting of arrays, therefore the array is loaded into memory here. This might fail for arrays that don't fit into memory."
    )

    permutation_idxs = np.argsort(data, kind="mergesort", axis=axis)
    if not asc:  # [::-1] not possible
        permutation_idxs = np.flip(
            permutation_idxs
        )  # descending - the order is flipped

    if nodata is None:  # ignore np.nan values
        if len(data.shape) > 1:
            raise ValueError(
                "order with nodata=None is not supported for arrays with more than one dimension, as this would result in sparse multi-dimensional arrays."
            )
        # sort the original data first, to get correct position of no data values
        sorted_data = np.take_along_axis(data, permutation_idxs, axis=axis)
        return permutation_idxs[~pd.isnull(sorted_data)]
    elif nodata is False:  # put location/index of np.nan values first
        # sort the original data first, to get correct position of no data values
        sorted_data = data[permutation_idxs]
        return np.append(
            permutation_idxs[pd.isnull(sorted_data)],
            permutation_idxs[~pd.isnull(sorted_data)],
        )
    elif nodata is True:  # default argsort behaviour, np.nan values are put last
        return permutation_idxs


def rearrange(
    data: ArrayLike,
    order: ArrayLike,
    axis: Optional[int] = None,
    source_transposed_axis: int = None,
):
    if len(data) == 0:
        return data
    if isinstance(data, list):
        data = np.asarray(data)
    if len(data.shape) == 1 and axis is None:
        axis = 0
    if isinstance(order, list):
        order = np.asarray(order)
    if len(order.shape) != 1:
        raise ValueError(
            f"rearrange: order must be one-dimensional, but has {len(order.shape)} dimensions. "
        )
    return np.take(data, indices=order, axis=axis)


def sort(
    data: ArrayLike,
    asc: Optional[bool] = True,
    nodata: Optional[bool] = None,
    axis: Optional[int] = None,
):
    if isinstance(data, list):
        data = np.asarray(data)
    if len(data) == 0:
        return data
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
