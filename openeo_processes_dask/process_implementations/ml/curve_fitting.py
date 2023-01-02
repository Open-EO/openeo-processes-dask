from typing import Callable, Optional

import dask.array as da
import numpy as np
import xarray as xr
from scipy import optimize

from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["fit_curve", "predict_curve"]


def fit_curve(data: RasterCube, parameters: list, function: Callable, dimension: str):
    data = data.fillna(0)  # zero values (masked) are not considered
    if dimension in [
        "time",
        "t",
        "times",
    ]:  # time dimension must be converted into values
        dates = data[dimension].values
        timestep = [
            ((x - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")) for x in dates
        ]
        step = np.array(timestep)
        data[dimension] = step
    else:
        step = dimension

    if isinstance(parameters, xr.DataArray):
        apply_f = lambda x, y, p: optimize.curve_fit(
            function, x[np.nonzero(y)], y[np.nonzero(y)], p
        )[0]
        in_dims = [[dimension], [dimension], ["params"]]
        add_arg = [step, data, parameters]
        output_size = len(parameters["params"])
    else:
        apply_f = lambda x, y: optimize.curve_fit(
            function, x[np.nonzero(y)], y[np.nonzero(y)], parameters
        )[0]
        in_dims = [[dimension], [dimension]]
        add_arg = [step, data]
        output_size = len(parameters)
    values = xr.apply_ufunc(
        apply_f,
        *add_arg,
        vectorize=True,
        input_core_dims=in_dims,
        output_core_dims=[["params"]],
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {"params": output_size},
        }
    )
    values["params"] = list(range(len(values["params"])))
    values.attrs = data.attrs
    return values


def predict_curve(
    data: RasterCube,
    parameters: RasterCube,
    function: Callable,
    dimension: str,
    labels=None,
):
    data = data.fillna(0)
    if type(labels) in [int, float, str]:
        labels = [labels]
    test = labels
    if dimension in [
        "time",
        "t",
        "times",
    ]:  # time dimension must be converted into values
        dates = data[dimension].values
        if test is None:
            timestep = [
                (
                    (np.datetime64(x) - np.datetime64("1970-01-01"))
                    / np.timedelta64(1, "s")
                )
                for x in dates
            ]
            labels = np.array(timestep)
        else:
            coords = labels
            labels = [
                (
                    (np.datetime64(x) - np.datetime64("1970-01-01"))
                    / np.timedelta64(1, "s")
                )
                for x in labels
            ]
            labels = np.array(labels)
    else:
        if test is None:
            labels = data[dimension].values
        else:
            coords = labels
    values = xr.apply_ufunc(
        lambda a: function(labels, *a),
        parameters,
        vectorize=True,
        input_core_dims=[["params"]],
        output_core_dims=[[dimension]],
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {dimension: len(labels)},
        },
    )
    if test is None:
        values = values.transpose(*data.dims)
        values[dimension] = data[dimension]
        predicted = data.where(data != 0, values)
    else:
        predicted = values.transpose(*data.dims)
        predicted[dimension] = coords
    if dimension in ["t", "times"]:
        predicted = predicted.rename({dimension: "time"})
    predicted.attrs = data.attrs
    return predicted
