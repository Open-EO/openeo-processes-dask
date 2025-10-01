import shutil

import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask.process_implementations.cubes.load import load_stac, load_url
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(10, 10, 10, 5)])
@pytest.mark.parametrize("dtype", [np.float32])
# @pytest.mark.parametrize("dims", [("x", "y", "t", "bands")])


def test_load_stac(bounding_box, random_raster_data, temporal_interval):
    url = "./tests/data/stac/s2_l2a_test_item.json"
    output_cube = load_stac(
        url=url,
        spatial_extent=bounding_box,
        bands=["red"],
    )

    assert output_cube.openeo is not None
    assert len(output_cube[output_cube.openeo.x_dim]) > 0
    assert len(output_cube[output_cube.openeo.y_dim]) > 0
    assert len(output_cube[output_cube.openeo.band_dims[0]]) > 0
    assert len(output_cube[output_cube.openeo.temporal_dims[0]]) > 0

    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08", "SCL"],
        backend="dask",
    )

    input_cube.to_dataset(dim="bands").to_zarr("./tests/data/s2_l2a_zarr_sample.zarr")

    url = "./tests/data/stac/s2_l2a_zarr_sample.json"
    output_cube = load_stac(
        url=url,
        bands=["B04"],
    )

    assert output_cube.openeo is not None
    assert len(output_cube[output_cube.openeo.x_dim]) > 0
    assert len(output_cube[output_cube.openeo.y_dim]) > 0
    assert len(output_cube[output_cube.openeo.band_dims[0]]) > 0
    assert len(output_cube[output_cube.openeo.temporal_dims[0]]) > 0
    shutil.rmtree("./tests/data/s2_l2a_zarr_sample.zarr")


def test_load_url():
    URL = "https://github.com/ValentinaHutter/polygons/raw/master/geoparquet/example%20file.geoparquet"

    load_vector = load_url(url=URL, format="Parquet")

    assert isinstance(load_vector, xr.Dataset)

    URL = "https://raw.githubusercontent.com/ValentinaHutter/polygons/master/polygons_all.json"

    load_vector = load_url(url=URL, format="GeoJSON")

    assert isinstance(load_vector, xr.Dataset)
