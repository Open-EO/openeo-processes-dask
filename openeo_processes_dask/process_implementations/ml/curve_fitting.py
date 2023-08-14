from typing import Callable, Optional

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from openeo_processes_dask.process_implementations.cubes import apply_dimension
from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
)

__all__ = ["fit_curve", "predict_curve"]


def fit_curve(data: RasterCube, parameters: list, function: Callable, dimension: str):
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )

    # In the spec, parameters is a list, but xr.curvefit requires names for them,
    # so we do this to generate names locally
    parameters = {f"param_{i}": v for i, v in enumerate(parameters)}

    # The dimension along which to predict cannot be chunked!
    rechunked_data = data.chunk({dimension: -1})

    def wrapper(f):
        def _wrap(*args, **kwargs):
            return f(
                *args,
                **kwargs,
                positional_parameters={"x": 0, "parameters": slice(1, None)},
            )

        return _wrap

    # .curvefit returns some extra information that isn't required by the OpenEO process
    # so we simply drop these here.
    fit_result = (
        rechunked_data.curvefit(
            dimension,
            wrapper(function),
            p0=parameters,
            param_names=list(parameters.keys()),
        )
        .drop_dims(["cov_i", "cov_j"])
        .to_array()
        .squeeze()
    )
    return fit_result


def predict_curve(
    parameters: RasterCube,
    function: Callable,
    dimension: str,
    labels: Optional[ArrayLike] = None,
):
    # parameters: (x, y, bands, param)
    if np.issubdtype(labels.dtype, np.datetime64):
        labels = labels.astype(int)

    def wrapper(f):
        def _wrap(*args, **kwargs):
            return f(
                *args,
                positional_parameters={"parameters": 0},
                named_parameters={"x": labels},
                **kwargs,
            )

        return _wrap

    return xr.apply_ufunc(
        wrapper(function),
        parameters,
        vectorize=True,
        input_core_dims=[["param"]],
        output_core_dims=[[dimension]],
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {dimension: len(labels)},
        },
    )
