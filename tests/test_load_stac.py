import pytest

from openeo_processes_dask.process_implementations.cubes.load import load_stac


def test_load_stac(temporal_interval, bounding_box):
    url = (
        "https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2"
    )
    output_cube = load_stac(
        url=url,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["red"],
    )

    assert output_cube.openeo is not None
    assert len(output_cube[output_cube.openeo.x_dim]) > 0
    assert len(output_cube[output_cube.openeo.y_dim]) > 0
    assert len(output_cube[output_cube.openeo.temporal_dims[0]]) > 0
