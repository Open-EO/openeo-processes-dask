from functools import partial

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.comparison import between, eq, neq
from openeo_processes_dask.process_implementations.cubes.apply import apply
from openeo_processes_dask.process_implementations.cubes.reduce import reduce_dimension
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


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
    if expected:
        assert between(x, min, max, exclude_max)
        assert between(np.array([x]), min, max, exclude_max)
        assert between(da.from_array(np.array([x])), min, max, exclude_max)
    else:
        assert not between(x, min, max, exclude_max)
        assert not between(np.array([x]), min, max, exclude_max)
        assert not between(da.from_array(np.array([x])), min, max, exclude_max)


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
        process_registry["is_nan"], x=ParameterReference(from_parameter="x")
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
        process_registry["is_valid"], x=ParameterReference(from_parameter="x")
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
        process_registry["is_infinite"], x=ParameterReference(from_parameter="x")
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
        process_registry["eq"], y=0.5, x=ParameterReference(from_parameter="x")
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
        process_registry["neq"], y=0.5, x=ParameterReference(from_parameter="x")
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
        process_registry["gt"], x=ParameterReference(from_parameter="x"), y=0.5
    )
    output_cube_gt = apply(data=input_cube, process=_process)
    _process = partial(
        process_registry["gte"], x=ParameterReference(from_parameter="x"), y=0.5
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
        process_registry["lt"], x=ParameterReference(from_parameter="x"), y=0.5
    )
    output_cube_lt = apply(data=input_cube, process=_process)
    _process = partial(
        process_registry["lte"], x=ParameterReference(from_parameter="x"), y=0.5
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
        process_registry["between"],
        x=ParameterReference(from_parameter="x"),
        min=0.1,
        max=0.5,
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
    xr.testing.assert_equal(output_cube, xr.zeros_like(input_cube))
