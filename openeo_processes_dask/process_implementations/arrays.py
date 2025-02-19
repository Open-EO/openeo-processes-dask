import copy
import itertools
import logging
from typing import Any, Callable, Optional, Union

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from openeo_pg_parser_networkx.pg_schema import DateTime
from xarray.core.duck_array_ops import isnull, notnull

from openeo_processes_dask.process_implementations.comparison import is_valid
from openeo_processes_dask.process_implementations.cubes.utils import _is_dask_array
from openeo_processes_dask.process_implementations.exceptions import (
    ArrayElementNotAvailable,
    ArrayElementParameterConflict,
    ArrayElementParameterMissing,
    ArrayLabelConflict,
    ArrayLengthMismatch,
    ArrayNotLabeled,
    LabelExists,
    TooManyDimensions,
)

logger = logging.getLogger(__name__)


__all__ = [
    "array_element",
    "array_create",
    "array_create_labeled",
    "array_modify",
    "array_concat",
    "array_append",
    "array_contains",
    "array_find",
    "array_find_label",
    "array_filter",
    "array_labels",
    "array_apply",
    "array_interpolate_linear",
    "first",
    "last",
    "order",
    "rearrange",
    "sort",
    "count",
]


def get_labels(data, dimension="labels", axis=0, dim_labels=None):
    if isinstance(data, xr.DataArray):
        dimension = data.dims[0] if len(data.dims) == 1 else dimension
        if axis:
            dimension = data.dims[axis]
        labels = data[dimension].values
        data = data.values
    else:
        labels = []
        if isinstance(data, list):
            data = np.asarray(data)
    if not isinstance(dim_labels, type(None)):
        labels = dim_labels
    return labels, data


def array_element(
    data: ArrayLike,
    index: Optional[int] = None,
    label: Optional[str] = None,
    return_nodata: Optional[bool] = False,
    axis=None,
    context=None,
    dim_labels=None,
):
    if index is None and label is None:
        raise ArrayElementParameterMissing(
            "The process `array_element` requires either the `index` or `labels` parameter to be set."
        )

    if index is not None and label is not None:
        raise ArrayElementParameterConflict(
            "The process `array_element` only allows that either the `index` or the `labels` parameter is set."
        )
    dim_labels, data = get_labels(data, axis=axis, dim_labels=dim_labels)

    if label is not None:
        if len(dim_labels) == 0:
            raise ArrayNotLabeled(
                "The array is not a labeled array, but the `label` parameter is set. Use the `index` instead."
            )
        if isinstance(label, DateTime):
            label = label.to_numpy()
        (index,) = np.where(dim_labels == label)
        if len(index) == 0:
            index = None
        else:
            index = index[0]

    try:
        if index is not None:
            element = np.take(data, index, axis=axis)
            return element
        else:
            raise IndexError
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


def array_create_labeled(data: ArrayLike, labels: ArrayLike) -> ArrayLike:
    if isinstance(data, list):
        data = np.array(data)
    if len(data) == len(labels):
        data = xr.DataArray(data, dims=["labels"], coords={"labels": labels})
        return data
    raise ArrayLengthMismatch(
        "The number of values in the parameters `data` and `labels` don't match."
    )


def array_modify(
    data: ArrayLike, values: ArrayLike, index: int, length: Optional[int] = 1, axis=None
) -> ArrayLike:
    labels, data = get_labels(data, axis=axis)
    values_labels, values = get_labels(values, axis=axis)

    if index > len(data):
        raise ArrayElementNotAvailable(
            "The array can't be modified as the given index is larger than the number of elements in the array."
        )
    if len(np.intersect1d(labels, values_labels)) > 0:
        raise ArrayLabelConflict(
            "At least one label exists in both arrays and the conflict must be resolved before."
        )

    def modify(data):
        first = data[:index]
        modified = np.append(first, values)
        if index + length < len(data):
            modified = np.append(modified, data[index + length :])
        return modified

    if axis:
        if _is_dask_array(data):
            if data.size > 50000000:
                raise Exception(
                    f"Cannot load data of shape: {data.shape} into memory. "
                )
            # currently, there seems to be no way around loading the values,
            # apply_along_axis cannot handle dask arrays
            data = data.compute()
        modified = np.apply_along_axis(modify, axis=axis, arr=data)
    else:
        modified = modify(data)

    if len(labels) > 0:
        first = labels[:index]
        modified_labels = np.append(first, values_labels)
        if index + length < len(labels):
            modified_labels = np.append(modified_labels, labels[index + length :])
        modified = array_create_labeled(data=modified, labels=modified_labels)

    return modified


def array_concat(array1: ArrayLike, array2: ArrayLike, axis=None) -> ArrayLike:
    labels1, array1 = get_labels(array1)
    labels2, array2 = get_labels(array2)

    if len(np.intersect1d(labels1, labels2)) > 0:
        raise ArrayLabelConflict(
            "At least one label exists in both arrays and the conflict must be resolved before."
        )

    if (len(array1.shape) - len(array2.shape)) == 1:
        if axis is None:
            s1 = np.array(list(array1.shape))
            s2 = list(array2.shape)
            s2.append(0)
            s2 = np.array(s2)

            axis = np.argmax(s1 != s2)

        array2 = np.expand_dims(array2, axis=axis)

    if axis:
        concat = np.concatenate([array1, array2], axis=axis)
    else:
        concat = np.concatenate([array1, array2])

    # e.g. concating int32 and str arrays results in the result being cast to a Unicode dtype of a certain length (e.g. <U22).
    # There isn't really anything better to do as numpy does not support heterogenuous arrays.
    # Best we can do at this point is to at least make the user aware that this is what has happened.
    if array1.dtype.kind != array2.dtype.kind:
        logger.warning(
            f"array_concat: different datatypes for array1 ({array1.dtype}) and array2 ({array2.dtype}), cast to {concat.dtype}"
        )

    if len(labels1) > 0 and len(labels2) > 0:
        labels = np.concatenate([labels1, labels2])
        concat = array_create_labeled(data=concat, labels=labels)
    return concat


def array_append(
    data: ArrayLike,
    value: Any,
    label: Optional[Any] = None,
    dim_labels=None,
    axis=None,
) -> ArrayLike:
    if axis:
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if (isinstance(value, np.ndarray) or isinstance(value, da.core.Array)) and len(
            value.flatten()
        ) == 1:
            value = value.flatten()[0]

        value = np.take(np.ones_like(data), indices=0, axis=axis) * value
        concat = array_concat(data, value, axis=axis)
        return concat

    if dim_labels:
        data = array_create_labeled(data=data, labels=dim_labels)
    if label is not None:
        labels, _ = get_labels(data)
        if label in labels:
            raise LabelExists(
                "An array element with the specified label already exists."
            )
        value = array_create_labeled(data=[value], labels=[label])
        return array_concat(data, value)

    if (
        not isinstance(value, list)
        and not isinstance(value, np.ndarray)
        and not isinstance(value, da.core.Array)
    ):
        value = [value]

    return array_concat(data, value)


def array_contains(data: ArrayLike, value: Any, axis=None) -> bool:
    # TODO: Contrary to the process spec, our implementation does interpret temporal strings before checking them here
    # This is somewhat implicit in how we currently parse parameters, so cannot be easily changed.
    labels, data = get_labels(data)
    value_is_valid = False
    valid_dtypes = [np.number, np.bool_, np.str_]
    for dtype in valid_dtypes:
        if np.issubdtype(type(value), dtype):
            value_is_valid = True
    if len(np.shape(data)) != 1 and axis is None:
        return False
    if not value_is_valid or pd.isnull(value):
        return False
    else:
        return np.isin(data, value).any(axis=axis)


def array_find(
    data: ArrayLike,
    value: Any,
    reverse: Optional[bool] = False,
    axis: Optional[int] = None,
) -> np.number:
    labels, data = get_labels(data, axis)

    if reverse:
        data = np.flip(data, axis=axis)

    idxs = (data == value).argmax(axis=axis)

    mask = ~np.array((data == value).any(axis=axis))
    if not isinstance(value, str) and np.isnan(value):
        mask = True
    if reverse:
        if axis is None:
            size = data.size
        else:
            size = data.shape[axis]
        idxs = size - 1 - idxs

    logger.warning(
        "array_find: numpy has no sentinel value for missing data in integer arrays, therefore np.masked_array is used to return the indices of found elements. Further operations might fail if not defined for masked arrays."
    )
    if isinstance(idxs, da.Array):
        idxs = idxs.compute_chunk_sizes()
        masked_idxs = np.atleast_1d(da.ma.masked_array(idxs, mask=mask))
    else:
        masked_idxs = np.atleast_1d(np.ma.masked_array(idxs, mask=mask))

    return masked_idxs


def array_find_label(data: ArrayLike, label: Union[str, int, float], dim_labels=None):
    if dim_labels:
        labels = dim_labels
    else:
        labels, data = get_labels(data)
    if len(labels) > 0:
        return array_find(labels, label)
    return None


def array_filter(
    data: ArrayLike, condition: Callable, context: Optional[Any] = None, axis=None
) -> ArrayLike:
    labels, data = get_labels(data, axis=axis)
    if not context:
        context = {}
    positional_parameters = {"x": 0}
    named_parameters = {"x": data, "context": context}
    if callable(condition):
        process_to_apply = np.vectorize(condition)
        filtered_data = process_to_apply(
            data,
            positional_parameters=positional_parameters,
            named_parameters=named_parameters,
        )
        if len(np.shape(data)) == 1:
            data = data[filtered_data.astype(bool)]
        else:
            if axis:
                n_axis = len(np.shape(data))
                for ax in range(n_axis - 1, -1, -1):
                    if ax != axis:
                        filtered_data = filtered_data.astype(bool).all(axis=ax)
                filtered_data = np.argwhere(filtered_data).flatten()
                data = np.take(data, filtered_data, axis=axis)
                return data
        if len(labels) > 0:
            labels = labels[filtered_data]
            data = array_create_labeled(data, labels)
        return data
    raise Exception(f"Array could not be filtered as condition is not callable. ")


def array_labels(data: ArrayLike, axis=None, dim_labels=None) -> ArrayLike:
    if dim_labels:
        return dim_labels
    if isinstance(data, xr.DataArray) and axis:
        dim = data.dims[axis]
        labels, data = get_labels(data, dim)
    else:
        labels, data = get_labels(data)
    if len(labels) > 0:
        return labels
    if len(np.shape(data)) > 1:
        if axis:
            return np.arange(data.shape[axis])
        raise TooManyDimensions("array_labels is only implemented for 1D arrays.")
    return np.arange(len(data))


def array_apply(
    data: ArrayLike, process: Callable, context: Optional[Any] = None
) -> ArrayLike:
    labels, data = get_labels(data)
    if not context:
        context = {}
    positional_parameters = {"x": 0}
    named_parameters = {"x": data, "context": context}
    if callable(process):
        process_to_apply = np.vectorize(process)
        return process_to_apply(
            data,
            positional_parameters=positional_parameters,
            named_parameters=named_parameters,
        )
    raise Exception(f"Could not apply process as it is not callable. ")


def array_interpolate_linear(data: ArrayLike, axis=None, dim_labels=None):
    return_label = False
    x, data = get_labels(data, axis=axis)
    if len(x) > 0:
        dim_labels = x
        return_label = True
    if dim_labels:
        x = np.array(dim_labels)
    if np.array(x).dtype.type is np.str_:
        try:
            x = np.array(x, dtype="datetime64").astype(float)
        except Exception:
            try:
                x = np.array(x, dtype=float)
            except Exception:
                x = np.arange(len(data))
    if len(x) == 0:
        if axis:
            x = np.arange(data.shape[axis])
        else:
            x = np.arange(len(data))

    def interp(data):
        valid = np.isfinite(data)
        if (valid == 1).all():
            return data
        if len(x[valid]) < 2:
            return data
        data[~valid] = np.interp(
            x[~valid], x[valid], data[valid], left=np.nan, right=np.nan
        )

        return data

    if axis:
        if _is_dask_array(data):
            if data.size > 50000000:
                raise Exception(
                    f"Cannot load data of shape: {data.shape} into memory. "
                )
            # currently, there seems to be no way around loading the values,
            # apply_along_axis cannot handle dask arrays
            data = data.compute()
        data = np.apply_along_axis(interp, axis=axis, arr=data)
    else:
        data = interp(data)
    if return_label:
        return array_create_labeled(data=data, labels=dim_labels)
    return data


def first(
    data: ArrayLike,
    ignore_nodata: Optional[bool] = True,
    axis: Optional[str] = None,
):
    if isinstance(data, list):
        data = np.asarray(data)
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
    labels, data = get_labels(data)
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
    labels, data = get_labels(data)
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
    labels, data = get_labels(data)
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


def count(
    data: ArrayLike,
    condition: Optional[Union[Callable, bool]] = None,
    context: Any = None,
    axis=None,
    keepdims=False,
):
    labels, data = get_labels(data)
    if condition is None:
        valid = is_valid(data)
        return np.nansum(valid, axis=axis, keepdims=keepdims)
    if condition is True:
        return np.nansum(np.ones_like(data), axis=axis, keepdims=keepdims)
    if callable(condition):
        if not context:
            context = {}
        context.pop("x", None)
        count = condition(x=data, **context)
        return np.nansum(count, axis=axis, keepdims=keepdims)
    raise Exception(f"Could not count values as condition is not callable. ")
