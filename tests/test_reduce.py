from functools import partial

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.cubes.reduce import (
    reduce_dimension,
    reduce_spatial,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_reduce_rqa(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    from openeo_processes_dask.process_implementations.arrays import array_apply
    from openeo_processes_dask.process_implementations.cubes.apply import (
        apply_dimension,
    )
    from openeo_processes_dask.process_implementations.experimental import (
        rqadeforestation,
    )

    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["rqa"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="t")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_reduce_dimension(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
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

    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="t")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )

    xr.testing.assert_equal(output_cube, input_cube.mean(dim="t"))


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_reduce_spatial(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["sum"].implementation,
        ignore_nodata=True,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube = reduce_spatial(data=input_cube, reducer=_process)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )

    xr.testing.assert_equal(output_cube, input_cube.sum(dim=["x", "y"]))
