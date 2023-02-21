import logging
from typing import Any, Optional

import dask.array as da
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from openeo_processes_dask.exceptions import (
    ArrayElementNotAvailable,
    ArrayElementParameterConflict,
    ArrayElementParameterMissing,
    TooManyDimensions,
)
from openeo_processes_dask.process_implementations.cubes.utils import _is_dask_array

logger = logging.getLogger(__name__)


__all__ = [
    "array_element",
    "array_create",
    "array_modify",
    "array_concat",
    "array_contains",
    "array_find",
    "array_labels",
    # "first",
    # "last",
    # "order",
    # "rearrange",
    # "sort",
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


def array_contains(data: ArrayLike, value: Any) -> bool:
    # TODO: Contrary to the process spec, our implementation does interpret temporal strings before checking them here
    # This is somewhat implicit in how we currently parse parameters, so cannot be easily changed.

    value_is_valid = False
    valid_dtypes = [np.number, np.bool_, np.str_]
    for dtype in valid_dtypes:
        if np.issubdtype(type(value), dtype):
            value_is_valid = True
    if not value_is_valid:
        return False

    if len(np.shape(data)) != 1:
        return False
    if pd.isnull(value):
        return np.isnan(data).any()
    else:
        return np.isin(data, value).any()


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
