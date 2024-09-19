from functools import partial

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.arrays import *
from openeo_processes_dask.process_implementations.cubes.reduce import reduce_dimension
from openeo_processes_dask.process_implementations.exceptions import (
    ArrayElementNotAvailable,
    TooManyDimensions,
)
from openeo_processes_dask.process_implementations.math import add
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_array_element(
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
        process_registry["array_element"].implementation,
        index=1,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="bands")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )

    xr.testing.assert_equal(output_cube, input_cube.isel({"bands": 1}, drop=True))

    # Use a label
    _process = partial(
        process_registry["array_element"].implementation,
        label="B02",
        data=ParameterReference(from_parameter="data"),
    )

    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="bands")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )

    xr.testing.assert_equal(output_cube, input_cube.loc[{"bands": "B02"}].drop("bands"))

    # When the index is out of range, we expect an ArrayElementNotAvailable exception to be thrown
    _process_not_available = partial(
        process_registry["array_element"].implementation,
        index=5,
        data=ParameterReference(from_parameter="data"),
    )

    with pytest.raises(ArrayElementNotAvailable):
        reduce_dimension(
            data=input_cube, reducer=_process_not_available, dimension="bands"
        )

        # When the index is out of range, we expect an ArrayElementNotAvailable exception to be thrown
    _process_no_data = partial(
        process_registry["array_element"].implementation,
        index=5,
        return_nodata=True,
        data=ParameterReference(from_parameter="data"),
    )

    output_cube_no_data_dask = reduce_dimension(
        data=input_cube, reducer=_process_no_data, dimension="bands"
    )
    nan_input_cube = input_cube.where(False, np.nan).isel({"bands": 0}, drop=True)
    assert isinstance(output_cube_no_data_dask.data, dask.array.Array)
    xr.testing.assert_equal(output_cube_no_data_dask, nan_input_cube)

    output_cube_no_data_numpy = reduce_dimension(
        data=input_cube.compute(), reducer=_process_no_data, dimension="bands"
    )
    assert isinstance(output_cube_no_data_numpy.data, np.ndarray)
    xr.testing.assert_equal(output_cube_no_data_dask, output_cube_no_data_numpy)


@pytest.mark.parametrize(
    "data, repeat",
    [([2], 3), ([], 10), ([1, 2, 3], 2), (["A", "B", "C"], 1), ([2, 1], 2)],
)
def test_array_create(data, repeat):
    result_np = array_create(data, repeat)
    np.testing.assert_array_equal(result_np, np.tile(data, repeat))
    assert len(result_np) == len(data) * repeat

    data_dask = da.from_array(data, chunks=-1)
    result_dask = array_create(data_dask, repeat)
    assert isinstance(result_dask, da.Array)
    np.testing.assert_array_equal(result_dask, result_np)


@pytest.mark.parametrize(
    "data, values, index, length, expected",
    [
        ([2, 3], [4, 7], 1, 0, [2, 4, 7, 3]),
        (["a", "d", "c"], ["b"], 1, 1, ["a", "b", "c"]),
        (["a", "c"], ["b"], 1, 0, ["a", "b", "c"]),
        ([np.nan, np.nan, "a", "b", "c"], [], 0, 2, ["a", "b", "c"]),
        (["a", "b", "c"], [], 1, 10, ["a"]),
    ],
)
def test_array_modify(data, values, index, length, expected):
    np.testing.assert_equal(array_modify(data, values, index, length), expected)
    np.testing.assert_equal(
        array_modify(np.array(data), values, index, length), expected
    )

    dask_result = array_modify(da.from_array(np.array(data)), values, index, length)
    assert isinstance(dask_result, da.Array)
    np.testing.assert_equal(dask_result.compute(), expected)


@pytest.mark.parametrize(
    "array1, array2, expected",
    [
        ([2, 3], [4, 7], [2, 3, 4, 7]),
        (["a", "b"], [1, 2], ["a", "b", 1, 2]),
    ],
)
def test_array_concat(array1, array2, expected):
    np.testing.assert_array_equal(array_concat(array1, array2), expected, strict=True)
    np.testing.assert_array_equal(
        array_concat(np.array(array1), np.array(array2)), expected, strict=True
    )

    dask_result = array_concat(
        da.from_array(np.array(array1)), da.from_array(np.array(array2))
    )
    np.testing.assert_array_equal(dask_result, np.array(expected), strict=True)


@pytest.mark.parametrize(
    "data, value, expected",
    [
        ([2, 3], 4, [2, 3, 4]),
        (["a", "b"], 1, ["a", "b", 1]),
    ],
)
def test_array_append(data, value, expected):
    np.testing.assert_array_equal(array_append(data, value), expected, strict=True)
    np.testing.assert_array_equal(
        array_append(np.array(data), np.array([value])), expected, strict=True
    )
    dask_result = array_append(
        da.from_array(np.array(data)), da.from_array(np.array([value]))
    )
    np.testing.assert_array_equal(dask_result, np.array(expected), strict=True)


@pytest.mark.parametrize(
    "data, value, expected",
    [
        ([1, 2, 3], 2, True),
        (["A", "B", "C"], "b", False),
        ([1, 2, 3], "2", False),
        ([1, 2, np.nan], np.nan, False),
        ([[2, 1], [3, 4]], [1, 2], False),
        ([[2, 1], [3, 4]], 2, False),
        ([1, 2, 3], np.int64(2), True),
        ([1.1, 2.2, 3.3], np.float64(2.2), True),
        ([True, False, False], np.bool_(True), True),
    ],
)
def test_array_contains(data, value, expected):
    assert array_contains(data, value) == expected
    assert array_contains(np.array(data), value) == expected

    dask_result = array_contains(da.from_array(np.array(data)), value)
    assert dask_result == expected or dask_result.compute() == expected


def test_array_contains_axis():
    data = np.array([[4, 5, 6], [5, 7, 9]])

    result_0 = array_contains(data, 5, axis=0)
    np.testing.assert_array_equal(result_0, np.array([True, True, False]))

    result_1 = array_contains(data, 5, axis=1)
    np.testing.assert_array_equal(result_1, np.array([True, True]))


def test_array_contains_object_dtype():
    assert not array_contains([{"a": "b"}, {"c": "d"}], {"a": "b"})
    assert not array_contains(np.array([{"a": "b"}, {"c": "d"}]), {"a": "b"})

    # Dask doesn't understand the `object` dtype and will error if encountered
    with pytest.raises(NotImplementedError):
        assert not array_contains(
            da.from_array(np.array([{"a": "b"}, {"c": "d"}])), {"a": "b"}
        )


@pytest.mark.parametrize(
    "data, value, expected, axis, reverse",
    [
        ([1, 0, 3, 2], 3, 2, None, False),
        ([1, 0, 3, 2, np.nan, 3], np.nan, 999999, None, False),
        ([1, 0, 3, 0, 2], 0, 1, None, False),
        ([[1, 0, 3, 2], [5, 3, 6, 8]], 3, [999999, 1, 0, 999999], 0, False),
        ([[1, 0, 3, 2], [5, 3, 6, 8]], 3, [2, 1], 1, False),
        ([1, 0, 3, 2], 3, 2, None, True),
        ([1, 0, 3, 2, np.nan, 3], np.nan, 999999, None, True),
        ([1, 0, 3, 0, 2], 0, 3, None, True),
        ([[1, 0, 3, 2], [5, 3, 6, 8]], 3, [999999, 1, 0, 999999], 0, True),
        ([[1, 0, 3, 2], [5, 3, 6, 8]], 3, [2, 1], 1, True),
        (["A", "B", "C"], "b", 99999, None, False),
    ],
)
def test_array_find(data, value, expected, axis, reverse):
    result_list = array_find(data, value=value, axis=axis, reverse=reverse)
    result_np = array_find(np.array(data), value=value, axis=axis, reverse=reverse)
    result_dask = array_find(
        da.from_array(np.array(data)), value, axis=axis, reverse=reverse
    )

    np.testing.assert_array_equal(result_list, expected)
    np.testing.assert_array_equal(result_np, expected)
    assert isinstance(result_dask, da.Array)
    np.testing.assert_array_equal(result_dask, expected)


def test_array_labels():
    """Tests `array_labels` function."""
    np.testing.assert_array_equal(array_labels([1, 0, 3, 2]), [0, 1, 2, 3])
    with pytest.raises(TooManyDimensions):
        array_labels(np.array([[1, 0, 3, 2], [5, 0, 6, 4]]))


def test_array_apply(process_registry):
    _process = partial(
        process_registry["add"].implementation,
        y=1,
        x=ParameterReference(from_parameter="x"),
    )

    output_cube = array_apply(data=np.array([1, 2, 3, 4, 5, 6]), process=_process)
    assert (output_cube == [2, 3, 4, 5, 6, 7]).all()


@pytest.mark.parametrize(
    "data, expected",
    [
        ([np.nan, 1, np.nan, 6, np.nan, -8], [np.nan, 1, 3.5, 6, -1, -8]),
        ([np.nan, 1, np.nan, np.nan], [np.nan, 1, np.nan, np.nan]),
    ],
)
def test_array_interpolate_linear(data, expected):
    assert np.array_equal(
        array_interpolate_linear(data),
        expected,
        equal_nan=True,
    )
    data_np = np.array(data)
    assert np.array_equal(
        array_interpolate_linear(data_np),
        expected,
        equal_nan=True,
    )
    data_da = da.from_array(data_np)
    assert np.array_equal(
        array_interpolate_linear(data_da),
        expected,
        equal_nan=True,
    )


def test_first():
    assert first(np.array([1, 0, 3, 2])) == 1
    assert pd.isnull(first(np.array([np.nan, 2, 3]), ignore_nodata=False))
    assert first(np.array([np.nan, 2, 3]), ignore_nodata=True) == 2
    assert pd.isnull(first([]))


def test_first_along_axis():
    multi_axis_array = np.array([[1, 0, 3, 2], [np.nan, 6, 7, 9]])
    expected_result_0_true = np.array([1, 0, 3, 2])
    expected_result_1_true = np.array([1, 6])
    expected_result_0_false = np.array([1, 0, 3, 2])
    expected_result_1_false = np.array([1, np.nan])

    assert np.array_equal(
        first(multi_axis_array, ignore_nodata=True, axis=0),
        expected_result_0_true,
        equal_nan=True,
    )
    assert np.array_equal(
        first(multi_axis_array, ignore_nodata=True, axis=1),
        expected_result_1_true,
        equal_nan=True,
    )
    assert np.array_equal(
        first(multi_axis_array, ignore_nodata=False, axis=0),
        expected_result_0_false,
        equal_nan=True,
    )
    assert np.array_equal(
        first(multi_axis_array, ignore_nodata=False, axis=1),
        expected_result_1_false,
        equal_nan=True,
    )


def test_last():
    assert last([1, 0, 3, 2]) == 2
    assert pd.isnull(last([0, 1, np.nan], ignore_nodata=False))
    assert last([0, 1, np.nan], ignore_nodata=True) == 1
    assert pd.isnull(last([]))


@pytest.mark.parametrize(
    "data, asc, nodata, expected",
    [
        (
            [6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9],
            True,
            None,
            [1, 2, 8, 5, 0, 4, 7, 9, 10],
        ),
        (
            [6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9],
            True,
            True,
            [1, 2, 8, 5, 0, 4, 7, 9, 10, 3, 6],
        ),
        (
            [6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9],
            False,
            True,
            [6, 3, 10, 9, 7, 4, 0, 5, 8, 2, 1],
        ),
        (
            [6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9],
            False,
            False,
            [6, 3, 10, 9, 7, 4, 0, 5, 8, 2, 1],
        ),
    ],
)
def test_order(data, asc, nodata, expected):
    np.testing.assert_array_equal(order(data=data, asc=asc, nodata=nodata), expected)
    np.testing.assert_array_equal(
        order(data=np.array(data), asc=asc, nodata=nodata), np.array(expected)
    )
    np.testing.assert_array_equal(
        order(data=da.from_array(np.array(data)), asc=asc, nodata=nodata),
        da.from_array(np.array(expected)),
    )


@pytest.mark.parametrize(
    "data, order, axis, expected",
    [
        ([5, 4, 3], [2, 1, 0], None, [3, 4, 5]),
        ([5, 4, 3, 2], [0, 2, 1, 3], 0, [5, 3, 4, 2]),
        ([5, 4, 3, 2], [1, 3], 0, [4, 2]),
        ([[5, 4, 3, 2], [5, 4, 3, 2]], [1, 3], 1, [[4, 2], [4, 2]]),
    ],
)
def test_rearrange(data, order, axis, expected):
    np.testing.assert_array_equal(
        rearrange(data=data, order=order, axis=axis), expected
    )
    np.testing.assert_array_equal(
        rearrange(data=np.array(data), order=order, axis=axis), np.array(expected)
    )
    np.testing.assert_array_equal(
        rearrange(data=da.from_array(np.array(data)), order=order, axis=axis),
        da.from_array(np.array(expected)),
    )


def test_rearrange_mismatched_shape():
    with pytest.raises(ValueError):
        rearrange(data=[[5, 4, 3, 2], [5, 4, 3, 2]], order=[[]], axis=1)


@pytest.mark.parametrize(
    "data, asc, nodata, expected",
    [
        (
            [6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9],
            True,
            None,
            [-1, 2, 3, 4, 6, 7, 8, 9, 9],
        ),
        (
            [6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9],
            False,
            True,
            [9, 9, 8, 7, 6, 4, 3, 2, -1, np.nan, np.nan],
        ),
    ],
)
def test_sort(data, asc, nodata, expected):
    """Tests `sort` function."""
    assert np.isclose(
        sort(data=data, asc=asc, nodata=nodata),
        expected,
        equal_nan=True,
    ).all()
    assert np.isclose(
        sort(data=np.array(data), asc=asc, nodata=nodata),
        expected,
        equal_nan=True,
    ).all()
    assert np.isclose(
        sort(data=da.from_array(np.array(data)), asc=asc, nodata=nodata),
        expected,
        equal_nan=True,
    ).all()


@pytest.mark.parametrize("size", [(3, 3, 2, 4)])
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

    input_cube[:, :, :, 0] = 1
    _process = partial(
        process_registry["array_find"].implementation,
        data=ParameterReference(from_parameter="data"),
        value=1,
        reverse=False,
    )
    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="bands")
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert output_cube.dims == ("x", "y", "t")
    xr.testing.assert_equal(output_cube, xr.zeros_like(output_cube))

    _process = partial(
        process_registry["first"].implementation,
        data=ParameterReference(from_parameter="data"),
        ignore_nodata=True,
    )
    input_cube[0, :, :, :2] = np.nan
    input_cube[0, :, :, 2] = 1
    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="bands")
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert output_cube.dims == ("x", "y", "t")
    xr.testing.assert_equal(output_cube, xr.ones_like(output_cube))

    input_cube[0, 0, 0, 0] = 99999
    _process = partial(
        process_registry["array_contains"].implementation,
        data=ParameterReference(from_parameter="data"),
        value=99999,
    )
    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="bands")
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert output_cube[0, 0, 0].data.compute().item() is True
    assert not output_cube[slice(1, None), :, :].data.compute().any()


@pytest.mark.parametrize("size", [(3, 3, 2, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_count(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    _process = partial(
        process_registry["count"].implementation,
        data=ParameterReference(from_parameter="data"),
    )
    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="bands")
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert output_cube.dims == ("x", "y", "t")
    xr.testing.assert_equal(output_cube, xr.zeros_like(output_cube) + 4)

    _process = partial(
        process_registry["count"].implementation,
        data=ParameterReference(from_parameter="data"),
        condition=True,
    )
    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="bands")
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert output_cube.dims == ("x", "y", "t")
    xr.testing.assert_equal(output_cube, xr.zeros_like(output_cube) + 4)

    _process = partial(
        process_registry["count"].implementation,
        data=ParameterReference(from_parameter="data"),
        condition=process_registry["gt"].implementation,
    )
    output_cube = reduce_dimension(
        data=input_cube,
        reducer=_process,
        dimension="bands",
        context={"y": -100},
    )
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert output_cube.dims == ("x", "y", "t")
    xr.testing.assert_equal(output_cube, xr.zeros_like(output_cube) + 4)

    _process = partial(
        process_registry["count"].implementation,
        data=ParameterReference(from_parameter="data"),
        condition=process_registry["is_infinite"].implementation,
    )
    output_cube = reduce_dimension(
        data=input_cube,
        reducer=_process,
        dimension="bands",
    )
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert output_cube.dims == ("x", "y", "t")
    xr.testing.assert_equal(output_cube, xr.zeros_like(output_cube))
