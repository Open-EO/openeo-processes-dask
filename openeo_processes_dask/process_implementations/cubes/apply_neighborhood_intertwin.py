from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import xarray as xr

from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["apply_neighborhood_intertwin"]


def apply_neighborhood_intertwin(
    data: RasterCube,
    process: Callable,
    size: dict[int],
    overlap: Optional[dict[int]] = None,
    context: Optional[dict] = None,
) -> RasterCube:
    positional_parameters = {"data": 0}
    named_parameters = {"context": context}

    stride = {**size}
    if overlap:
        size, stride = update_size_and_stride_with_overlap(size, overlap)

    new_dim_names = {i: f"window_{i}" for i in data.dims[::-1] if i != "bands"}
    window_data = data.rolling(size, center=True).construct(
        new_dim_names, stride=stride
    )
    reduced_data = window_data.reduce(
        process,
        dim=tuple(new_dim_names.values()),
        keep_attrs=True,
        positional_parameters=positional_parameters,
        named_parameters=named_parameters,
    )

    return reduced_data


def update_size_and_stride_with_overlap(size, overlap):
    size = {k: s + overlap[k] * 2 for k, s in size.items()}
    stride = {k: s - overlap[k] for k, s in size.items()}
    return size, stride
