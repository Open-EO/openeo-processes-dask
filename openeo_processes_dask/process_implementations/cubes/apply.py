from typing import Callable, Optional

import numpy as np
import xarray as xr

from openeo_processes_dask.exceptions import DimensionNotAvailable
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["apply", "apply_dimension"]


def apply(
    data: RasterCube, process: Callable, context: Optional[dict] = None, **kwargs
) -> RasterCube:
    positional_parameters = {"x": 0}
    named_parameters = {"context": context}
    result = xr.apply_ufunc(
        process,
        data,
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
        },
    )
    return result


def apply_dimension(
    data: RasterCube,
    process: Callable,
    dimension: str,
    target_dimension: Optional[str] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> RasterCube:
    if context is None:
        context = {}

    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    if target_dimension is None:
        target_dimension = dimension

    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    result = xr.apply_ufunc(
        process,
        data,
        input_core_dims=[[dimension]],
        output_core_dims=[[target_dimension]],
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
        },
    )

    transposed_result = result.transpose(*data.dims)
    if not np.array_equal(
        transposed_result.coords[target_dimension].data, data.coords[dimension].data
    ):
        transposed_result = transposed_result.assign_coords(
            {
                target_dimension: np.arange(
                    len(transposed_result.coords[target_dimension].data)
                )
            }
        )

    return transposed_result
