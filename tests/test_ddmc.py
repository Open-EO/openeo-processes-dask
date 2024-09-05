from functools import partial

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import (
    BoundingBox,
    ParameterReference,
    TemporalInterval,
)

from openeo_processes_dask.process_implementations.cubes.load import load_stac
from openeo_processes_dask.process_implementations.cubes.reduce import (
    reduce_dimension,
    reduce_spatial,
)
from openeo_processes_dask.process_implementations.exceptions import (
    ArrayElementNotAvailable,
)
from openeo_processes_dask.process_implementations.experimental.ddmc import ddmc
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 20, 5)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_ddmc_instance_dims(
    temporal_interval: TemporalInterval, bounding_box: BoundingBox, random_raster_data
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["nir08", "nir09", "cirrus", "swir16", "swir22"],
        backend="dask",
    )

    data = ddmc(input_cube)

    assert isinstance(data, xr.DataArray)
    assert set(input_cube.dims) == set(data.dims)


@pytest.mark.parametrize("size", [(30, 30, 20, 5)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_ddmc_target_band(
    temporal_interval: TemporalInterval, bounding_box: BoundingBox, random_raster_data
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["nir08", "nir09", "cirrus", "swir16", "swir22"],
        backend="dask",
    )

    data_band = ddmc(data=input_cube, target_band="ddmc")
    assert "ddmc" in data_band.dims


@pytest.mark.parametrize("size", [(30, 30, 20, 5)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_ddmc_input_cube_exception(
    temporal_interval: TemporalInterval, bounding_box: BoundingBox, random_raster_data
):
    input_cube_exception = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["b04", "nir09", "cirrus", "swir16", "swir22"],
        backend="dask",
    )

    with pytest.raises(KeyError):
        data = ddmc(input_cube_exception)
