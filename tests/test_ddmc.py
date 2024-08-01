from functools import partial

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.cubes.load import load_stac
from openeo_processes_dask.process_implementations.cubes.reduce import (
    reduce_dimension,
    reduce_spatial,
)
from openeo_processes_dask.process_implementations.ddmc import ddmc
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 20, 5)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_ddmc(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["nir08", "nir09", "cirrus", "swir16", "swir22"],
        backend="dask",
    )

    data = ddmc(input_cube)

    assert isinstance(data, xr.DataArray)
    assert input_cube.dims == data.dims

    data = ddmc(input_cube, target_band="ddmc")

    assert "ddmc" in data.dims

    input_cube_exception = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["b04", "nir09", "cirrus", "swir16", "swir22"],
        backend="dask",
    )

    with pytest.raises(ArrayElementNotAvailable):
        data = ddmc(input_cube_exception)
