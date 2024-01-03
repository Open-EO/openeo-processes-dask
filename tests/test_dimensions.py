import numpy as np
import pytest

from openeo_processes_dask.process_implementations.cubes.general import (
    add_dimension,
    drop_dimension,
)
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionLabelCountMismatch,
    DimensionNotAvailable,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_add_dimension(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    output_cube = add_dimension(data=input_cube, name="other", label="test")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        expected_dims=["x", "y", "t", "bands", "other"],
    )
    assert output_cube.openeo.band_dims[0] == "bands"
    assert output_cube.openeo.temporal_dims[0] == "t"
    assert output_cube.openeo.spatial_dims == ("x", "y")
    assert output_cube.openeo.other_dims[0] == "other"

    output_cube_2 = add_dimension(
        data=input_cube, name="weird", label="test", type="temporal"
    )
    assert output_cube_2.openeo.temporal_dims[1] == "weird"


@pytest.mark.parametrize("size", [(30, 30, 1, 2)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_drop_dimension(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B04"],
        backend="dask",
    )
    DIM_TO_DROP = "bands"

    with pytest.raises(DimensionNotAvailable):
        drop_dimension(input_cube, "notthere")

    with pytest.raises(DimensionLabelCountMismatch):
        drop_dimension(input_cube, DIM_TO_DROP)

    suitable_cube = input_cube.where(input_cube.bands == "B02", drop=True)

    output_cube = drop_dimension(suitable_cube, DIM_TO_DROP)
    DIMS_TO_KEEP = tuple(filter(lambda y: y != DIM_TO_DROP, input_cube.dims))
    assert DIM_TO_DROP not in output_cube.dims
    assert DIMS_TO_KEEP == output_cube.dims
