from functools import partial

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.ml.curve_fitting import (
    fit_curve,
    predict_curve,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_curve(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["multiply"],
        x=ParameterReference(from_parameter="data"),
        y=ParameterReference(from_parameter="parameters"),
    )

    fitted_cube = fit_curve(
        data=input_cube, parameters=[1], function=_process, dimension="t"
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=fitted_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert fitted_cube.dims == ("x", "y", "bands", "params")
    assert fitted_cube["params"].values == 0

    predicted_cube = predict_curve(
        data=input_cube, parameters=fitted_cube, function=_process, dimension="t"
    )
    general_output_checks(
        input_cube=input_cube,
        output_cube=predicted_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert predicted_cube.dims == ("x", "y", "t", "bands")
    assert (predicted_cube["t"].values == input_cube["t"].values).all()
