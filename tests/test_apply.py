from functools import partial

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.cubes.apply import (
    apply,
    apply_dimension,
)
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["add"], y=1, x=ParameterReference(from_parameter="x")
    )

    output_cube = apply(data=input_cube, process=_process)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=(input_cube + 1),
    )

    xr.testing.assert_equal(output_cube, input_cube + 1)


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_case_1(
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
        process_registry["add"], y=1, x=ParameterReference(from_parameter="data")
    )

    # Target dimension is null and therefore defaults to the source dimension
    output_cube_same_pixels = apply_dimension(
        data=input_cube, process=_process, dimension="x"
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube_same_pixels,
        verify_attrs=True,
        verify_crs=True,
        expected_results=(input_cube + 1),
    )


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_reduce_dimension(
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
        process_registry["mean"], data=ParameterReference(from_parameter="data")
    )

    # Target dimension is null and therefore defaults to the source dimension
    output_cube_reduced = apply_dimension(
        data=input_cube, process=_process, dimension="x"
    )

    expected_output = (input_cube.mean(dim="x")).expand_dims("x")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube_reduced,
        verify_attrs=True,
        verify_crs=False,
        expected_results=expected_output,
    )


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_target_dimension(
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
        process_registry["mean"], data=ParameterReference(from_parameter="data")
    )

    # Target dimension is null and therefore defaults to the source dimension
    output_cube_reduced = apply_dimension(
        data=input_cube, process=_process, dimension="x", target_dimension="target"
    )

    expected_output = (input_cube.mean(dim="x")).expand_dims("target")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube_reduced,
        verify_attrs=True,
        verify_crs=False,
        expected_results=expected_output,
    )

    # Target dimension is null and therefore defaults to the source dimension
    output_cube_reduced = apply_dimension(
        data=input_cube, process=_process, dimension="x", target_dimension="y"
    )

    expected_output = (
        (input_cube.mean(dim="x"))
        .expand_dims("target")
        .drop("y")
        .rename({"target": "y"})
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube_reduced,
        verify_attrs=True,
        verify_crs=False,
        expected_results=expected_output,
    )
