from functools import partial

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.cubes.apply_neighborhood_intertwin import (
    apply_neighborhood_intertwin,
)


def test_empty_data_input_raises_exception(process_registry):
    with pytest.raises(ValueError):
        input_cube = make_full_datacube((0, 0), 1)
        _process = make_sum_process(process_registry)
        apply_neighborhood_intertwin(
            data=input_cube, process=_process, size=dict(x=3, y=3)
        ).compute()


def make_sum_process(process_registry):
    _process = partial(
        process_registry["sum"].implementation,
        ignore_nodata=True,
        data=ParameterReference(from_parameter="data"),
    )

    return _process


def make_full_datacube(shape, value):
    data = da.full(shape, value).astype(np.float32)
    input_cube = xr.DataArray(data, dims=("x", "y"))
    return input_cube


def test_some_single_pixel_datacube(process_registry):
    input_cube = make_full_datacube((1, 1), 1)
    sum_proc = make_sum_process(process_registry)
    output_cube = apply_neighborhood_intertwin(
        data=input_cube, process=sum_proc, size=dict(x=1, y=1)
    ).compute()
    assert_datacube_eq(output_cube, make_full_datacube((1, 1), 1))


def assert_datacube_eq(actual, expected):
    xr.testing.assert_equal(actual, expected)


def test_by_default_rolls_with_window_size_as_stride(process_registry):
    input_cube = make_full_datacube((5, 5), 1)
    sum_proc = make_sum_process(process_registry)
    output_cube = apply_neighborhood_intertwin(
        input_cube, sum_proc, size=dict(x=2, y=2)
    ).compute()
    assert_datacube_eq(
        output_cube, make_datacube([[1.0, 2.0, 2.0], [2.0, 4.0, 4.0], [2.0, 4.0, 4.0]])
    )


def make_datacube(values):
    return xr.DataArray(da.from_array(values), dims=("x", "y"))


def test_truncates_when_window_size_does_not_fit_exactly(process_registry):
    input_cube = make_full_datacube((4, 4), 1)
    sum_proc = make_sum_process(process_registry)
    output_cube = apply_neighborhood_intertwin(
        input_cube, sum_proc, size=dict(x=2, y=2)
    ).compute()
    assert_datacube_eq(output_cube, make_datacube([[1.0, 2.0], [2.0, 4.0]]))


def test_overlap_extends_to_neighbors_on_both_side_with_stride_specified_by_size(
    process_registry,
):
    input_cube = make_full_datacube((4, 4), 1)
    sum_proc = make_sum_process(process_registry)
    output_cube = apply_neighborhood_intertwin(
        input_cube, sum_proc, size=dict(x=1, y=1), overlap=dict(x=1, y=1)
    ).compute()
    assert_datacube_eq(output_cube, make_datacube([[4.0, 6.0], [6.0, 9.0]]))


def test_non_square_window(process_registry):
    input_cube = make_full_datacube((5, 5), 1)
    sum_proc = make_sum_process(process_registry)
    output_cube = apply_neighborhood_intertwin(
        input_cube, sum_proc, size=dict(x=1, y=2)
    ).compute()
    assert_datacube_eq(
        output_cube,
        make_datacube(
            [
                [1.0, 2.0, 2.0],
                [1.0, 2.0, 2.0],
                [1.0, 2.0, 2.0],
                [1.0, 2.0, 2.0],
                [1.0, 2.0, 2.0],
            ]
        ),
    )


def test_non_square_overlap(process_registry):
    input_cube = make_full_datacube((4, 4), 1)
    sum_proc = make_sum_process(process_registry)
    output_cube = apply_neighborhood_intertwin(
        input_cube, sum_proc, size=dict(x=1, y=1), overlap=dict(x=1, y=2)
    ).compute()
    assert_datacube_eq(output_cube, make_datacube([[6.0, 6.0], [9.0, 9.0]]))


def test_zero_size_results_in_stride_one(process_registry):
    input_cube = make_full_datacube((4, 4), 1)
    sum_proc = make_sum_process(process_registry)
    output_cube = apply_neighborhood_intertwin(
        input_cube, sum_proc, size=dict(x=0, y=0), overlap=dict(x=1, y=1)
    ).compute()
    assert_datacube_eq(
        output_cube,
        make_datacube(
            [
                [1.0, 2.0, 2.0, 2.0],
                [2.0, 4.0, 4.0, 4.0],
                [2.0, 4.0, 4.0, 4.0],
                [2.0, 4.0, 4.0, 4.0],
            ]
        ),
    )


def test_negative_size_results_in_stride_one(process_registry):
    input_cube = make_full_datacube((4, 4), 1)
    sum_proc = make_sum_process(process_registry)
    output_cube = apply_neighborhood_intertwin(
        input_cube, sum_proc, size=dict(x=-1, y=-1), overlap=dict(x=2, y=2)
    ).compute()
    assert_datacube_eq(
        output_cube,
        make_datacube(
            [
                [4.0, 6.0, 6.0, 4.0],
                [6.0, 9.0, 9.0, 6.0],
                [6.0, 9.0, 9.0, 6.0],
                [4.0, 6.0, 6.0, 4.0],
            ]
        ),
    )
