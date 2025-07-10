import pytest
import xarray as xr

from openeo_processes_dask.process_implementations.cubes.load import load_stac, load_url


def test_load_stac(bounding_box):
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

    url = "https://stac.openeo.eurac.edu/api/v1/pgstac/collections/s2_l2a_zarr_sample"
    output_cube = load_stac(
        url=url,
        bands=["B04"],
    )

    assert output_cube.openeo is not None
    assert len(output_cube[output_cube.openeo.x_dim]) > 0
    assert len(output_cube[output_cube.openeo.y_dim]) > 0
    assert len(output_cube[output_cube.openeo.band_dims[0]]) > 0
    assert len(output_cube[output_cube.openeo.temporal_dims[0]]) > 0


def test_load_url():
    URL = "https://github.com/ValentinaHutter/polygons/raw/master/geoparquet/example%20file.geoparquet"

    load_vector = load_url(url=URL, format="Parquet")

    assert isinstance(load_vector, xr.Dataset)

    URL = "https://raw.githubusercontent.com/ValentinaHutter/polygons/master/polygons_all.json"

    load_vector = load_url(url=URL, format="GeoJSON")

    assert isinstance(load_vector, xr.Dataset)
