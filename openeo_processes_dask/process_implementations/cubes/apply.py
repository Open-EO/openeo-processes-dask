from typing import Callable, Optional

import xarray as xr

from openeo_processes_dask.exceptions import DimensionNotAvailable
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["apply"]


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
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    parameters = {"data": data, "context": context}

    output_data = process(parameters=parameters, dimension=dimension)
    raise NotImplementedError("apply_dimension isn't supported yet!")
