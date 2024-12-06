from functools import partial

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import resolution_from_affine
from openeo_pg_parser_networkx.pg_schema import ParameterReference, TemporalInterval
from pyproj.crs import CRS

from openeo_processes_dask.process_implementations.cubes.reduce import reduce_dimension
from openeo_processes_dask.process_implementations.cubes.resample import (
    resample_cube_spatial,
    resample_cube_temporal,
    resample_spatial,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize(
    "output_crs",
    [
        3587,
        "32633",
        "+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs",
        "4326",
    ],
)
@pytest.mark.parametrize("output_res", [5, 30, 60])
@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("dims", [("x", "y", "bands"), ("x", "y", "t"), ("x", "y")])
def test_resample_spatial(
    output_crs,
    output_res,
    temporal_interval,
    bounding_box,
    random_raster_data,
    dims,
    process_registry,
):
    """Test to ensure resolution gets changed correctly."""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["mean"].implementation,
        ignore_nodata=True,
        data=ParameterReference(from_parameter="data"),
    )

    if "bands" not in dims:
        output_cube = reduce_dimension(
            data=input_cube, reducer=_process, dimension="bands"
        )

    if "t" not in dims:
        output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="t")

    with pytest.raises(Exception):
        output_cube = resample_spatial(
            data=input_cube, projection=output_crs, resolution=output_res, method="bad"
        )

    output_cube = resample_spatial(
        data=input_cube, projection=output_crs, resolution=output_res
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=False,
    )

    assert output_cube.odc.spatial_dims == ("y", "x")
    assert output_cube.rio.crs == CRS.from_user_input(output_crs)

    if output_crs != "4326":
        assert resolution_from_affine(output_cube.odc.geobox.affine).x == output_res
        assert resolution_from_affine(output_cube.odc.geobox.affine).y == -output_res

    if output_cube.rio.crs.is_geographic:
        assert min(output_cube.x) >= -180
        assert max(output_cube.x) <= 180

        assert min(output_cube.y) >= -90
        assert max(output_cube.y) <= 90


@pytest.mark.parametrize(
    "output_crs",
    [
        3587,
        "32633",
        "+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs",
        "4326",
    ],
)
@pytest.mark.parametrize("output_res", [5, 30, 60])
@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_resample_cube_spatial(
    output_crs, output_res, temporal_interval, bounding_box, random_raster_data
):
    """Test to ensure resolution gets changed correctly."""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    resampled_cube = resample_spatial(
        data=input_cube, projection=output_crs, resolution=output_res
    )

    with pytest.raises(Exception):
        output_cube = resample_cube_spatial(
            data=input_cube, target=resampled_cube, method="bad"
        )

    output_cube = resample_cube_spatial(
        data=input_cube, target=resampled_cube, method="average"
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        expected_dims=input_cube.dims,
        verify_attrs=False,
        verify_crs=False,
    )

    assert output_cube.odc.spatial_dims == ("y", "x")


@pytest.mark.parametrize(
    "output_crs",
    [
        3587,
        "32633",
        "+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs",
    ],
)
@pytest.mark.parametrize("output_res", [5, 30, 60])
@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_resample_cube_spatial_small(
    output_crs, output_res, temporal_interval, bounding_box, random_raster_data
):
    """Test to ensure resolution gets changed correctly."""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    resampled_cube = resample_spatial(
        data=input_cube, projection=output_crs, resolution=output_res
    )

    output_cube = resample_cube_spatial(
        data=input_cube, target=resampled_cube[10:60, 20:150, :, :], method="average"
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        expected_dims=input_cube.dims,
        verify_attrs=False,
        verify_crs=False,
    )

    assert list(output_cube.shape) == list(resampled_cube.shape)
    assert (output_cube["x"].values == resampled_cube["x"].values).all()
    assert (output_cube["y"].values == resampled_cube["y"].values).all()


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize(
    "temporal_extent_1,temporal_extent_2",
    [
        (["2018-05-01", "2018-06-01"], ["2018-05-05", "2018-06-05"]),
        (["2019-01-01", "2019-03-01"], ["2019-02-01", "2019-03-01"]),
    ],
)
def test_aggregate_temporal_period(
    temporal_extent_1,
    temporal_extent_2,
    bounding_box,
    random_raster_data,
    process_registry,
):
    """"""
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=TemporalInterval.parse_obj(temporal_extent_1),
        bands=["B02", "B03", "B04", "B08"],
    )

    target_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=TemporalInterval.parse_obj(temporal_extent_2),
        bands=["B02", "B03", "B04", "B08"],
    )

    output_cube = resample_cube_temporal(
        data=input_cube, target=target_cube, dimension=None, valid_within=None
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )

    assert len(output_cube.t) == len(target_cube.t)
    assert (output_cube.t.values == target_cube.t.values).all()

    output_cube = resample_cube_temporal(
        data=input_cube, target=target_cube, dimension="t", valid_within=2
    )

    assert len(output_cube.t) == len(target_cube.t)
    assert (output_cube.t.values == target_cube.t.values).all()

    with pytest.raises(Exception):
        resample_cube_temporal(
            data=input_cube, target=target_cube, dimension="time", valid_within=None
        )
