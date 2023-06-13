import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import TemporalInterval

from openeo_processes_dask.process_implementations.cubes._filter import (
    filter_bands,
    filter_bbox,
    filter_temporal,
)
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionNotAvailable,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 30, 1)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_temporal(temporal_interval, bounding_box, random_raster_data):
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

    with pytest.raises(DimensionNotAvailable):
        filter_temporal(
            data=input_cube, extent=temporal_interval_part, dimension="immissing"
        )

    temporal_interval_open = TemporalInterval.parse_obj([None, "2018-05-03T00:00:00"])
    output_cube = filter_temporal(data=input_cube, extent=temporal_interval_open)

    xr.testing.assert_equal(
        output_cube,
        input_cube.loc[dict(t=slice("2018-05-01T00:00:00", "2018-05-02T23:59:59"))],
    )


@pytest.mark.parametrize("size", [(1, 1, 1, 2)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_bands(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "SCL"],
        backend="dask",
    )

    output_cube = filter_bands(data=input_cube, bands=["SCL"])

    assert output_cube["bands"].values == "SCL"


@pytest.mark.parametrize("size", [(30, 30, 1, 1)])
@pytest.mark.parametrize("dtype", [np.uint8])
def test_filter_bbox(
    temporal_interval, bounding_box, random_raster_data, bounding_box_small
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02"],
        backend="dask",
    )

    output_cube = filter_bbox(data=input_cube, extent=bounding_box_small)

    assert len(output_cube.y) < len(input_cube.y)
    assert len(output_cube.x) < len(input_cube.x)
