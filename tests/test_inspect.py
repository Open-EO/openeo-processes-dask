import logging

import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask.process_implementations.inspect import inspect
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_inspect_results(temporal_interval, bounding_box, random_raster_data):
    raster_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
    )

    assert inspect(np.zeros(1)) == np.zeros(1)
    xr.testing.assert_equal(raster_cube, inspect(raster_cube))


def test_inspect_logs(caplog):
    with caplog.at_level(logging.INFO):
        result = inspect(data=1, message="First log", code="Logging")
        assert result == 1
        assert "Logging: First log 1." in caplog.text

    with caplog.at_level(logging.WARNING):
        result = inspect(data=1, message="First log", level="warning")
        assert result == 1
        assert "User: First log 1." in caplog.text

    with caplog.at_level(logging.ERROR):
        result = inspect(data=1, level="error")
        assert result == 1
        assert "User:  1." in caplog.text

    with caplog.at_level(logging.DEBUG):
        result = inspect(data=1, level="debug")
        assert result == 1
        assert "User:  1." in caplog.text
