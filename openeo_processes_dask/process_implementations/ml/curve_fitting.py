from typing import Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from openeo_processes_dask.process_implementations.cubes import apply_dimension
from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
)

__all__ = ["fit_curve", "predict_curve"]


def fit_curve(
    data: RasterCube,
    parameters: list,
    function: Callable,
    dimension: str,
    ignore_nodata: bool = True,
):
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"Provided dimension ({dimension}) not found in data.dims: {data.dims}"
        )
    bands_required = False
    if "bands" in data.dims:
        if len(data["bands"].values) == 1:
            bands_required = data["bands"].values[0]

    try:
        # Try parsing as datetime first
        dates = data[dimension].values
        dates = np.asarray(dates, dtype=np.datetime64)
    except ValueError:
        dates = np.asarray(data[dimension].values)

    if np.issubdtype(dates.dtype, np.datetime64):
        timestep = [
            (
                (np.datetime64(x) - np.datetime64("1970-01-01", "s"))
                / np.timedelta64(1, "s")
            )
            for x in dates
        ]
        data[dimension] = np.array(timestep)

    dims_before = list(data.dims)

    # In the spec, parameters is a list, but xr.curvefit requires names for them,
    # so we do this to generate names locally
    parameters = {f"param_{i}": v for i, v in enumerate(parameters)}

    chunking = {key: "auto" for key in data.dims if key != dimension}
    chunking[dimension] = -1

    # The dimension along which to fit the curves cannot be chunked!
    rechunked_data = data.chunk(chunking)
    rechunked_data = rechunked_data.persist()

    def wrapper(f):
        def _wrap(*args, **kwargs):
            return f(
                *args,
                **kwargs,
                positional_parameters={"x": 0, "parameters": slice(1, None)},
            )

        return _wrap

    expected_dims_after = list(dims_before)
    expected_dims_after[dims_before.index(dimension)] = "param"

    # .curvefit returns some extra information that isn't required by the OpenEO process
    # so we simply drop these here.
    fit_result = (
        rechunked_data.curvefit(
            dimension,
            wrapper(function),
            p0=parameters,
            param_names=list(parameters.keys()),
            skipna=ignore_nodata,
        )
        .drop_dims(["cov_i", "cov_j"])
        .to_array()
        .squeeze()
    )

    fit_result.attrs = data.attrs
    fit_result = fit_result.rio.write_crs(rechunked_data.rio.crs)
    if bands_required and not "bands" in fit_result.dims:
        fit_result = fit_result.assign_coords(**{"bands": bands_required})
        fit_result = fit_result.expand_dims(dim="bands")

    fit_result = fit_result.transpose(*expected_dims_after)

    return fit_result


def predict_curve(
    parameters: RasterCube,
    function: Callable,
    dimension: str,
    labels: ArrayLike,
    data: RasterCube = None,
):
    labels_were_datetime = False
    dims_before = list(parameters.dims)
    initial_labels = labels

    try:
        # Try parsing as datetime first
        labels = np.asarray(labels, dtype=np.datetime64)
    except ValueError:
        labels = np.asarray(labels)

    if np.issubdtype(labels.dtype, np.datetime64):
        labels_were_datetime = True
        timestep = [
            (
                (np.datetime64(x) - np.datetime64("1970-01-01", "s"))
                / np.timedelta64(1, "s")
            )
            for x in labels
        ]
        labels = np.array(timestep)

    # This is necessary to pipe the arguments correctly through @process
    def wrapper(f):
        def _wrap(*args, **kwargs):
            return f(
                *args,
                positional_parameters={"parameters": 0},
                named_parameters={"x": labels},
                **kwargs,
            )

        return _wrap

    expected_dims_after = list(dims_before)
    expected_dims_after[dims_before.index("param")] = dimension

    predictions = xr.apply_ufunc(
        wrapper(function),
        parameters,
        vectorize=True,
        input_core_dims=[["param"]],
        output_core_dims=[[dimension]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {dimension: len(labels)},
        },
    ).transpose(*expected_dims_after)

    predictions = predictions.assign_coords({dimension: labels.data})

    if labels_were_datetime:
        predictions[dimension] = initial_labels

    return predictions
