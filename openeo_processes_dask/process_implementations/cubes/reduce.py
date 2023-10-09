from typing import Callable, Optional

import numpy as np

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
)

__all__ = ["reduce_dimension", "reduce_spatial"]


def reduce_dimension(
    data: RasterCube,
    reducer: Callable,
    dimension: str,
    context: Optional[dict] = None,
) -> RasterCube:
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    reduced_data = data.reduce(
        reducer,
        dim=dimension,
        keep_attrs=True,
        positional_parameters=positional_parameters,
        named_parameters=named_parameters,
    )

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
    data: RasterCube, reducer: Callable, context: Optional[dict] = None
) -> RasterCube:
    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    spatial_dims = data.openeo.spatial_dims if data.openeo.spatial_dims else None
    return data.reduce(
        reducer,
        dim=spatial_dims,
        keep_attrs=True,
        positional_parameters=positional_parameters,
        named_parameters=named_parameters,
    )
