from functools import partial

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.comparison import (
    between,
    eq,
    is_valid,
    neq,
)
from openeo_processes_dask.process_implementations.cubes.apply import apply
from openeo_processes_dask.process_implementations.cubes.reduce import reduce_dimension
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, True),
        (np.nan, False),
        (np.array([1, np.nan]), np.array([True, False])),
        ({"test": "ok"}, True),
        ([1, 2, np.nan], np.array([True, True, False])),
    ],
)
@pytest.mark.parametrize("is_dask", [True, False])
def test_is_valid(value, expected, is_dask):
    value = np.asarray(value)

    if is_dask:
        value = da.from_array(value)

    output = is_valid(value)
    np.testing.assert_array_equal(output, expected)

    if is_dask:
        assert hasattr(output, "dask")


@pytest.mark.parametrize(
    "x, y, delta, case_sensitive",
    [
        (1, 1, None, True),
        (-1, -1.001, 0.01, None),
        (115, 110, 10, None),
        ("Test", "test", None, False),
    ],
)
def test_eq(x, y, delta, case_sensitive):
    assert eq(x=x, y=y, delta=delta, case_sensitive=case_sensitive)
    assert eq(
        x=np.array([x]), y=np.array([y]), delta=delta, case_sensitive=case_sensitive
    )
    assert eq(
        x=da.from_array(np.array([x])),
        y=da.from_array(np.array([y])),
        delta=delta,
        case_sensitive=case_sensitive,
    )


@pytest.mark.parametrize(
    "x, y, delta, case_sensitive",
    [(1, "1", None, True), (1.02, 1, 0.01, True), ("Test", "test", None, True)],
)
def test_neq(x, y, delta, case_sensitive):
    assert neq(x=x, y=y, delta=delta, case_sensitive=case_sensitive)
    assert neq(
        x=np.array([x]), y=np.array([y]), delta=delta, case_sensitive=case_sensitive
    )
    assert neq(
        x=da.from_array(np.array([x])),
        y=da.from_array(np.array([y])),
        delta=delta,
        case_sensitive=case_sensitive,
    )


@pytest.mark.parametrize(
    "x, min, max, exclude_max, expected",
    [
        (1, 0, 1, False, True),
        (1, 0, 1, True, False),
        (0.5, 1, 0, False, False),
        (-0.5, -1, 0, False, True),
    ],
)
def test_between(x, min, max, exclude_max, expected):
    assert between(x, min, max, exclude_max) == expected
    assert between(np.array([x]), min, max, exclude_max) == expected
    assert between(da.from_array(np.array([x])), min, max, exclude_max) == expected


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_is(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["is_valid"].implementation,
        x=ParameterReference(from_parameter="x"),
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    xr.testing.assert_equal(output_cube, xr.ones_like(input_cube))

    _process = partial(
        process_registry["is_infinite"].implementation,
        x=ParameterReference(from_parameter="x"),
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    xr.testing.assert_equal(output_cube, xr.zeros_like(input_cube))


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_compare(temporal_interval, bounding_box, random_raster_data, process_registry):
    # TODO: Add test with merge_cubes
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["eq"].implementation,
        y=200,
        x=ParameterReference(from_parameter="x"),
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    xr.testing.assert_equal(output_cube, xr.zeros_like(input_cube))

    _process = partial(
        process_registry["neq"].implementation,
        y=200,
        x=ParameterReference(from_parameter="x"),
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    xr.testing.assert_equal(output_cube, xr.ones_like(input_cube))

    _process = partial(
        process_registry["gt"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=200,
    )
    output_cube_gt = apply(data=input_cube, process=_process)
    _process = partial(
        process_registry["gte"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=200,
    )
    output_cube_gte = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    xr.testing.assert_equal(output_cube_gt, output_cube_gte)

    _process = partial(
        process_registry["lt"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=200,
    )
    output_cube_lt = apply(data=input_cube, process=_process)
    _process = partial(
        process_registry["lte"].implementation,
        x=ParameterReference(from_parameter="x"),
        y=200,
    )
    output_cube_lte = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    xr.testing.assert_equal(output_cube_lt, output_cube_lte)

    _process = partial(
        process_registry["between"].implementation,
        x=ParameterReference(from_parameter="x"),
        min=200,
        max=300,
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    xr.testing.assert_equal(output_cube, xr.zeros_like(input_cube))
    _process = partial(
        process_registry["between"].implementation,
        x=ParameterReference(from_parameter="x"),
        min=200,
        max=300,
        exclude_max=True,
    )
    output_cube_b2 = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    xr.testing.assert_equal(output_cube, output_cube_b2)
