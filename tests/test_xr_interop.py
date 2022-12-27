import numpy as np
import pytest

from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_openeo_accessor(temporal_interval, bounding_box, random_raster_data):
    raster_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
    )

    assert raster_cube.openeo is not None
    assert raster_cube.openeo.x_dim == "x"
    assert raster_cube.openeo.y_dim == "y"
    assert raster_cube.openeo.temporal_dims[0] == "t"

    with pytest.raises(NotImplementedError):
        raster_cube.openeo.z_dim

    raster_cube = raster_cube.rename({"t": "month"})
    assert raster_cube.openeo.temporal_dims[0] == "month"

    raster_cube = raster_cube.rename({"month": "NotATimeDim"})
    assert not raster_cube.openeo.temporal_dims
