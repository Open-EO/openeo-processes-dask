from functools import partial

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.arrays import array_element
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_array_element(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    output_cube = array_element(data=input_cube, index=0, dimension="bands")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=(input_cube.sel(bands="B02")),
    )

    xr.testing.assert_equal(output_cube, input_cube.sel(bands="B02"))

    input_array = [0, 1, 2, 3, 4]

    output_array = array_element(data=input_array, index=4)

    assert output_array == 4
    assert output_array == array_element(data=input_array, index=-1)
