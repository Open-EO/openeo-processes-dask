import logging
from typing import Callable, Optional

import dask.array as da
import numpy as np
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


__all__ = ["array_element", "array_filter", "count"]


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
