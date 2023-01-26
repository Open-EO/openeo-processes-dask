from functools import partial

import dask
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.exceptions import ArrayElementNotAvailable
from openeo_processes_dask.process_implementations.cubes.reduce import reduce_dimension
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
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

    _process = partial(
        process_registry["array_element"],
        index=1,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="bands")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )

    xr.testing.assert_equal(output_cube, input_cube.isel({"bands": 1}, drop=True))

    # When the index is out of range, we expect an ArrayElementNotAvailable exception to be thrown
    _process_not_available = partial(
        process_registry["array_element"],
        index=5,
        data=ParameterReference(from_parameter="data"),
    )

    with pytest.raises(ArrayElementNotAvailable):
        reduce_dimension(
            data=input_cube, reducer=_process_not_available, dimension="bands"
        )

        # When the index is out of range, we expect an ArrayElementNotAvailable exception to be thrown
    _process_no_data = partial(
        process_registry["array_element"],
        index=5,
        return_nodata=True,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_no_data_dask = reduce_dimension(
        data=input_cube, reducer=_process_no_data, dimension="bands"
    )
    nan_input_cube = input_cube.where(False, np.nan).isel({"bands": 0}, drop=True)
    assert isinstance(output_cube_no_data_dask.data, dask.array.Array)
    xr.testing.assert_equal(output_cube_no_data_dask, nan_input_cube)

    output_cube_no_data_numpy = reduce_dimension(
        data=input_cube.compute(), reducer=_process_no_data, dimension="bands"
    )
    assert isinstance(output_cube_no_data_numpy.data, np.ndarray)
    xr.testing.assert_equal(output_cube_no_data_dask, output_cube_no_data_numpy)
