from typing import Callable, Optional, Union

import numpy as np
import scipy.ndimage
import xarray as xr

from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
    KernelDimensionsUneven,
)

__all__ = ["apply", "apply_dimension", "apply_kernel"]


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

    is_new_dim_added = target_dimension is not None

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
            "source_transposed_axis": data.get_axis_num(dimension),
            "context": context,
        },
        exclude_dims={dimension},
    )

    reordered_result = result.transpose(*data.dims, ...).rename(
        {dimension: target_dimension}
    )

    if len(reordered_result[target_dimension]) == 1:
        reordered_result[target_dimension] = ["0"]

    if data.rio.crs is not None:
        try:
            reordered_result.rio.write_crs(data.rio.crs, inplace=True)
        except ValueError:
            pass

    if is_new_dim_added:
        reordered_result.openeo.add_dim_type(name=target_dimension, type="other")

    return reordered_result


def apply_kernel(
    data: RasterCube,
    kernel: np.ndarray,
    factor: Optional[float] = 1,
    border: Union[float, str, None] = 0,
    replace_invalid: Optional[float] = 0,
) -> RasterCube:
    kernel = np.asarray(kernel)
    if any(dim % 2 == 0 for dim in kernel.shape):
        raise KernelDimensionsUneven(
            "Each dimension of the kernel must have an uneven number of elements."
        )

    def convolve(data, kernel, mode="constant", cval=0, fill_value=0):
        dims = ("y", "x")
        convolved = lambda data: scipy.ndimage.convolve(
            data, kernel, mode=mode, cval=cval
        )

        data_masked = data.fillna(fill_value)

        return xr.apply_ufunc(
            convolved,
            data_masked,
            vectorize=True,
            dask="parallelized",
            input_core_dims=[dims],
            output_core_dims=[dims],
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True},
        ).transpose(*data.dims)

    openeo_scipy_modes = {
        "replicate": "nearest",
        "reflect": "reflect",
        "reflect_pixel": "mirror",
        "wrap": "wrap",
    }
    if isinstance(border, int) or isinstance(border, float):
        mode = "constant"
        cval = border
    else:
        mode = openeo_scipy_modes[border]
        cval = 0

    return convolve(data, kernel, mode, cval, replace_invalid) * factor
