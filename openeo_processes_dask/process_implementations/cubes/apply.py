from typing import Callable, Optional

import xarray as xr

from openeo_processes_dask.exceptions import DimensionNotAvailable
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["apply"]


def apply(
    data: RasterCube, process: Callable, context: Optional[dict] = None, **kwargs
) -> RasterCube:
    return NotImplementedError(
        "apply doesn't currently work with the process implementations in math, etc. Need to migrate to apply_ufunc to enable this!"
    )


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
