from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.exceptions import DimensionNotAvailable
from typing import Callable, Dict, Optional

__all__ = ["apply"]


def apply(data: RasterCube, process: Callable, context: Optional[Dict]=None, **kwargs) -> RasterCube:
    parameters = {"x": data, "context": context}
    return process(parameters=parameters, **kwargs)


def apply_dimension(
    data: RasterCube, process: Callable, dimension: str, target_dimension: Optional[str]=None, context: Optional[Dict]=None, **kwargs
) -> RasterCube:
    if dimension not in data.dims:
        raise DimensionNotAvailable(f"Provided dimension not found in data.dims: {data.dims}")

    parameters = {"data": data, "context": context}

    output_data = process(parameters=parameters, dimension=dimension)
    raise NotImplementedError("apply_dimension isn't supported yet!")