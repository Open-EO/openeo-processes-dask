from typing import Callable, Optional

import numpy as np
import xarray as xr

from openeo_processes_dask.exceptions import DimensionNotAvailable
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["apply", "apply_dimension"]


def apply(
    data: RasterCube, process: Callable, context: Optional[dict] = None
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

    # This transpose (and back later) is needed because apply_ufunc automatically moves
    # input_core_dimensions to the last axes
    reordered_data = data.transpose(..., dimension)

    result = xr.apply_ufunc(
        process,
        reordered_data,
        input_core_dims=[[dimension]],
        output_core_dims=[[dimension]],
        dask="allowed",
        kwargs={
            "positional_parameters": positional_parameters,
            "named_parameters": named_parameters,
            "axis": reordered_data.get_axis_num(dimension),
            "keepdims": True,
        },
        exclude_dims={dimension},
    )

    reordered_result = result.transpose(*data.dims, ...).rename(
        {dimension: target_dimension}
    )

    if len(data[dimension]) == len(reordered_result[target_dimension]):
        reordered_result.rio.write_crs(data.rio.crs, inplace=True)

    return reordered_result
