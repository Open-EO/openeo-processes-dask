from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.exceptions import DimensionNotAvailable
import numpy as np
from typing import Callable, Dict, Optional

__all__ = ["reduce_dimension", "reduce_spatial"]


def reduce_dimension(
    data: RasterCube, reducer: Callable, dimension: str, context: Optional[Dict]=None, **kwargs
) -> RasterCube:
    if dimension not in data.dims:
        raise DimensionNotAvailable(f"Provided dimension not found in data.dims: {data.dims}")
    parameters = {"data": data, "context": context}
    reduced_data = reducer(parameters=parameters, dimension=dimension)
    
    # Preset
    if "reduced_dimensions_min_values" not in data.attrs:
        reduced_data.attrs["reduced_dimensions_min_values"] = {}
    try:        
        reduced_data.attrs["reduced_dimensions_min_values"][dimension] = data.coords[dimension].values.min()
    except np.core._exceptions.UFuncTypeError as e:
        reduced_data.attrs["reduced_dimensions_min_values"][dimension] = 0

    return reduced_data


def reduce_spatial(data: RasterCube, reducer: Callable, context: Optional[Dict]=None, **kwargs) -> RasterCube:
    parameters = {"data": data, "context": context}
    return reducer(parameters=parameters, dimension=["x", "y"])
