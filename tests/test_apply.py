from functools import partial

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.cubes.apply import (
    apply,
    apply_dimension,
    apply_kernel,
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
        process_registry["add"].implementation,
        y=1,
        x=ParameterReference(from_parameter="x"),
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
        process_registry["add"].implementation,
        y=1,
        x=ParameterReference(from_parameter="data"),
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
        process_registry["mean"].implementation,
        data=ParameterReference(from_parameter="data"),
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
        input_cube.mean(dim="x")
        .expand_dims("target")
        .drop_vars("y")
        .rename({"target": "y"})
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube_reduced,
        verify_attrs=True,
        verify_crs=False,
        expected_results=expected_output,
    )

    assert "y" in output_cube_reduced.openeo.other_dims


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_ordering_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process_order = partial(
        process_registry["order"].implementation,
        data=ParameterReference(from_parameter="data"),
        nodata=True,
    )

    output_cube_order = apply_dimension(
        data=input_cube,
        process=_process_order,
        dimension="x",
        target_dimension="target",
    )

    expected_output_order = np.argsort(input_cube.data, kind="mergesort", axis=0)

    np.testing.assert_array_equal(output_cube_order.data, expected_output_order)
    # This is to remind us that currently dask arrays don't support sorting and notify us should that change in a future version.
    assert isinstance(output_cube_order.data, np.ndarray)

    _process_rearrange = partial(
        process_registry["rearrange"].implementation,
        data=ParameterReference(from_parameter="data"),
        order=da.from_array(np.array([0, 1, 2, 3])),
    )

    output_cube_rearrange = apply_dimension(
        data=input_cube, process=_process_rearrange, dimension="x", target_dimension="x"
    )

    np.testing.assert_array_equal(output_cube_rearrange.dims, input_cube.dims)
    # This is to remind us that currently dask arrays don't support sorting and notify us should that change in a future version.
    assert isinstance(output_cube_rearrange.data, da.Array)

    _process_sort = partial(
        process_registry["sort"].implementation,
        data=ParameterReference(from_parameter="data"),
        nodata=True,
    )

    output_cube_sort = apply_dimension(
        data=input_cube, process=_process_sort, dimension="x", target_dimension="target"
    )

    expected_output_sort = np.sort(input_cube.data, axis=0)

    np.testing.assert_array_equal(output_cube_sort.data, expected_output_sort)
    # This is to remind us that currently dask arrays don't support sorting and notify us should that change in a future version.
    assert isinstance(output_cube_sort.data, np.ndarray)

    rearrange_by_expected_order = np.take_along_axis(
        input_cube.data, indices=expected_output_order, axis=0
    )

    np.testing.assert_array_equal(
        output_cube_sort.data, rearrange_by_expected_order.data
    )


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_quantile_processes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )
    probability = 4

    _process_quantile = partial(
        process_registry["quantiles"].implementation,
        data=ParameterReference(from_parameter="data"),
        probabilities=probability,
    )

    output_cube_quantile = apply_dimension(
        data=input_cube,
        process=_process_quantile,
        dimension="t",
    )
    assert output_cube_quantile.shape == (6, 5, probability - 1, 4)


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_kernel(temporal_interval, bounding_box, random_raster_data):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    # Following kernel should leave cube unchanged
    kernel = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    output_cube = apply_kernel(data=input_cube, kernel=kernel)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=input_cube,
    )

    xr.testing.assert_equal(output_cube, input_cube)


# TODO: testing cummin, cummax


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_cumsum_process(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process_cumsum = partial(
        process_registry["cumsum"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_cumsum = apply_dimension(
        data=input_cube,
        process=_process_cumsum,
        dimension="t",
    ).compute()

    original_abs_sum = np.sum(np.abs(input_cube.data))

    cumsum_total = np.sum(np.abs(output_cube_cumsum.data))

    assert cumsum_total >= original_abs_sum


@pytest.mark.parametrize("size", [(6, 5, 30, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply_dimension_cumproduct_process(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process_cumsum = partial(
        process_registry["cumproduct"].implementation,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_cumprod = apply_dimension(
        data=input_cube,
        process=_process_cumsum,
        dimension="t",
    ).compute()

    # TODO: Looking for better solution of following steps

    original_data = np.abs(input_cube.data)
    original_data[np.isnan(original_data)] = 0
    original_abs_prod = np.sum(original_data)

    cumprod_data = np.abs(output_cube_cumprod.data)
    cumprod_data[np.isnan(cumprod_data)] = 0
    cumprod_total = np.sum(cumprod_data)

    assert cumprod_total >= original_abs_prod
