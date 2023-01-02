from functools import partial

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.core import process_registry
from openeo_processes_dask.process_implementations.arrays import array_element
from openeo_processes_dask.process_implementations.cubes.general import dimension_labels
from openeo_processes_dask.process_implementations.math import *
from openeo_processes_dask.process_implementations.ml.curve_fitting import (
    fit_curve,
    predict_curve,
)
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube

timesteps = 150
spatial = 2
rang = np.linspace(0, 4 * np.pi, timesteps)
# define data with y = 0 + 1 * cos() + 0.5 *sin()
curve = (
    np.ones((spatial, timesteps)) * (np.cos(rang) + 0.5 * np.sin(rang))
    + np.random.rand(spatial, timesteps) * 0.1
)
xdata = xr.DataArray(
    curve,
    coords=[
        np.arange(spatial),
        pd.date_range("2018-01-01", "2020-01-01", periods=timesteps),
    ],
    dims=["x", "time"],
)


def func_oeop(x, *parameters):
    # function with cos and sin: a + b*cos(2*pi/31557600*x) + c*sin(2*pi/31557600*x)
    a = array_element(**{"data": parameters, "index": 0})
    b = array_element(**{"data": parameters, "index": 1})
    c = array_element(**{"data": parameters, "index": 2})
    # cos(2*pi/31557600*x)
    t1 = multiply(x=divide(x=multiply(x=2, y=pi()), y=31557600), y=x)
    cos1 = cos(**{"x": t1})
    # sin(2*pi/31557600*x)
    t2 = multiply(x=divide(x=multiply(x=2, y=pi()), y=31557600), y=x)
    sin1 = sin(**{"x": t2})
    # multiply and sum terms up
    m1 = multiply(**{"x": b, "y": cos1})
    m2 = multiply(**{"x": c, "y": sin1})
    sum1 = add(**{"x": a, "y": m1})
    sum2 = add(**{"x": sum1, "y": m2})
    return sum2


params = fit_curve(xdata, parameters=[1, 1, 1], function=func_oeop, dimension="time")
assert (
    np.isclose(params, [0, 1, 0.5], atol=0.3)
).all()  # output should be close to 0, 1, 0.5
params_2 = fit_curve(xdata, parameters=params, function=func_oeop, dimension="time")
assert (np.isclose(params_2, [0, 1, 0.5], atol=0.3)).all()
assert (np.isclose(params, params_2, atol=0.01)).all()

predicted = predict_curve(
    xdata,
    params,
    func_oeop,
    dimension="time",
    labels=pd.date_range("2002-01-01", periods=24, freq="M"),
)
assert (predicted < 1.8).all()
predicted = predict_curve(
    xdata,
    params,
    func_oeop,
    dimension="time",
    labels=pd.date_range("2018-01-01", "2020-01-01", periods=timesteps),
)
assert (np.isclose(xdata, predicted, atol=0.5)).all()
dim_times = dimension_labels(xdata, "time")
predicted_dim_labels = predict_curve(
    xdata, params, func_oeop, dimension="time", labels=dim_times
)
assert xdata.dims == predicted_dim_labels.dims
assert (predicted_dim_labels < 1.8).all()
predicted_str_list = predict_curve(
    xdata,
    params,
    func_oeop,
    dimension="time",
    labels=["2002-01-31 00:00", "2002-02-28"],
)
predicted_str = predict_curve(
    xdata, params, func_oeop, dimension="time", labels="2002-01-31 00:00"
)
xr.testing.assert_equal(predicted_str, predicted_str_list.isel(time=[0]))
