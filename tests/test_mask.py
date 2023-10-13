import numpy as np
import pytest
from openeo_pg_parser_networkx.pg_schema import TemporalInterval

from openeo_processes_dask.process_implementations.cubes.mask_polygon import (
    mask_polygon,
)
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 30, 1)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_mask_polygon(
    temporal_interval,
    bounding_box,
    random_raster_data,
    polygon_geometry_small,
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],
        backend="dask",
    )

    output_cube = mask_polygon(data=input_cube, geometries=polygon_geometry_small)

    assert np.isnan(output_cube).sum() > np.isnan(input_cube).sum()
    assert len(output_cube.y) == len(input_cube.y)
    assert len(output_cube.x) == len(input_cube.x)
