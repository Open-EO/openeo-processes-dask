from typing import Callable, Optional

import numpy as np

from openeo_processes_dask.exceptions import DimensionNotAvailable
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["reduce_dimension", "reduce_spatial"]


def reduce_dimension(
    data: RasterCube,
    reducer: Callable,
    dimension: str,
    context: Optional[dict] = None,
    **kwargs,
) -> RasterCube:
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    reduced_data = data.reduce(reducer, dim=dimension, keep_attrs=True, context=context)

    # Preset
    if "reduced_dimensions_min_values" not in data.attrs:
        reduced_data.attrs["reduced_dimensions_min_values"] = {}
    try:
        reduced_data.attrs["reduced_dimensions_min_values"][dimension] = data.coords[
            dimension
        ].values.min()
    except np.core._exceptions.UFuncTypeError as e:
        reduced_data.attrs["reduced_dimensions_min_values"][dimension] = 0

    return reduced_data


def reduce_spatial(
    data: RasterCube, reducer: Callable, context: Optional[dict] = None, **kwargs
) -> RasterCube:
    return data.reduce(
        reducer, dimension=data.openeo.spatial_dims, keep_attrs=True, context=context
    )
