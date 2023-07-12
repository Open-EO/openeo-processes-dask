import pytest

from openeo_processes_dask.process_implementations.cubes.load import load_stac


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
