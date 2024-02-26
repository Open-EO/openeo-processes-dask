from functools import partial

import numpy as np
import pytest
from openeo_pg_parser_networkx.pg_schema import ParameterReference, TemporalInterval

from openeo_processes_dask.process_implementations.cubes.mask import mask
from openeo_processes_dask.process_implementations.cubes.mask_polygon import (
    mask_polygon,
)
from openeo_processes_dask.process_implementations.cubes.reduce import (
    reduce_dimension,
    reduce_spatial,
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

    output_cube = mask_polygon(data=input_cube, mask=polygon_geometry_small)

    assert np.isnan(output_cube).sum() > np.isnan(input_cube).sum()
    assert len(output_cube.y) == len(input_cube.y)
    assert len(output_cube.x) == len(input_cube.x)


@pytest.mark.parametrize("size", [(30, 30, 20, 2)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_mask(
    temporal_interval,
    bounding_box,
    random_raster_data,
    process_registry,
):
    """Test to ensure resolution gets changed correctly."""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03"],
        backend="dask",
    )

    mask_cube = input_cube > 0
    output_cube = mask(data=input_cube, mask=mask_cube)

    assert np.isnan(output_cube).sum() > np.isnan(input_cube).sum()
    assert len(output_cube.y) == len(input_cube.y)
    assert len(output_cube.x) == len(input_cube.x)

    _process = partial(
        process_registry["max"].implementation,
        ignore_nodata=True,
        data=ParameterReference(from_parameter="data"),
    )

    mask_cube_no_x = reduce_dimension(data=mask_cube, dimension="x", reducer=_process)
    with pytest.raises(Exception):
        output_cube = mask(data=input_cube, mask=mask_cube_no_x)

    # Mask should work without bands
    mask_cube_no_bands = reduce_dimension(
        data=mask_cube, dimension="bands", reducer=_process
    )
    output_cube = mask(data=input_cube, mask=mask_cube_no_bands)

    # Mask should work without time
    mask_cube_no_time = reduce_dimension(
        data=mask_cube, dimension="t", reducer=_process
    )
    output_cube = mask(data=input_cube, mask=mask_cube_no_time)

    # Mask should work without time and bands
    mask_cube_no_time_bands = reduce_dimension(
        data=mask_cube_no_bands, dimension="t", reducer=_process
    )
    output_cube = mask(data=input_cube, mask=mask_cube_no_time_bands)
