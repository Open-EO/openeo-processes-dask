from functools import partial

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.cubes.general import add_dimension
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_add_dimension(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    output_cube = add_dimension(data=input_cube, name="other", label="test")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        expected_dims=["x", "y", "t", "bands", "other"],
    )
