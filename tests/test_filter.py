import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import TemporalInterval

from openeo_processes_dask.process_implementations.cubes._filter import filter_temporal
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 30, 1)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_temporal(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],
        backend="dask",
    )

    temporal_interval_part = TemporalInterval.parse_obj(
        ["2018-05-15T00:00:00", "2018-06-01T00:00:00"]
    )
    output_cube = filter_temporal(data=input_cube, extent=temporal_interval_part)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )

    xr.testing.assert_equal(
        output_cube,
        input_cube.loc[dict(t=slice("2018-05-15T00:00:00", "2018-05-31T23:59:59"))],
    )
