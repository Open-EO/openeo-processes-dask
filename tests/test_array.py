from functools import partial

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference, BoundingBox, TemporalInterval

from openeo_processes_dask.process_implementations.arrays import *
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_array_element(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    output_cube = array_element(data=input_cube, index=0, dimension="bands")

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=(input_cube.sel(bands="B02")),
    )

    xr.testing.assert_equal(output_cube, input_cube.sel(bands="B02"))

    input_array = [0, 1, 2, 3, 4]

    output_array = array_element(data=input_array, index=4)

    assert output_array == 4
    assert output_array == array_element(data=input_array, index=-1)

def test_array_create():
    assert (array_create(2,3) == np.array([2,2,2])).all()
    assert (len(array_create([])) == 0)
    assert (array_create(np.array([1, 2, 3]), repeat=2) == np.array([1,2,3,1,2,3])).all()
    assert (array_create(["A", "B", "C"]) == np.array(["A", "B", "C"])).all()
    assert len(array_create([np.nan, 1], 2)) == 4

def test_array_modify():
    assert (array_modify(np.array([2,3]), np.array([4,7]), 1, 0) == np.array([2,4,7,3])).all()
    assert (array_modify(data = ["a","d","c"], values = ["b"], index = 1) == ['a','b','c']).all()
    assert (array_modify(data = ["a","c"], values = ["b"], index = 1, length = 0) == ["a","b","c"]).all()
    assert (array_modify(data = [np.nan,np.nan,"a","b","c"], values = [], index = 0, length = 2) == ["a","b","c"]).all()
    assert (array_modify(data=["a", "b", "c"], values=[], index=1, length=10) == ["a"])

def test_array_concat():
    assert (array_concat(np.array([2, 3]), np.array([4, 7]))== np.array([2,3,4,7])).all()
    assert (array_concat(array1 = ["a","b"], array2 = [1,2]) == np.array(['a','b','1','2'], dtype=object)).all()

def test_array_contains():
    assert array_contains([1, 2, 3], value=2)
    assert not array_contains(["A", "B", "C"], value="b")
    assert not array_contains([1, 2, 3], value="2")
    assert array_contains([1, 2, 3], value=2)
    assert array_contains([1, 2, np.nan], value=np.nan)
    assert array_contains([[1, 2], [3, 4]], value=[1, 2])
    assert array_contains([[1, 2], [3, 4]], value=2)
    assert array_contains([{"a": "b"}, {"c": "d"}], value={"a": "b"})

def test_array_find():
    assert array_find([1, 0, 3, 2], value= 3) == 2
    assert np.isnan(array_find([1, 0, 3, 2, np.nan, 3], value = np.nan))
    data = np.array([[2, 8, 2, 4], [0, np.nan, 2, 2]])
    assert (array_find(data, value = 2, dimension = 1) == [0,2]).all()
    assert (array_find(data, value = 2, reverse = True, dimension = 1) == [2,3]).all()

def test_array_labels():
    """ Tests `array_labels` function. """
    assert (array_labels([1, 0, 3, 2]) == np.array([0, 1, 2, 3])).all()
    assert (array_labels([[1, 0, 3, 2]], dimension = 0) == np.array([0]))

def test_first():
    assert first([1, 0, 3, 2]) == 1
    assert np.isnan(first([np.nan, 2, 3], ignore_nodata=False))
    assert np.isnan(first([]))

    test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
    first_elem_ref = np.array([[[3., 2.], [1., 2.]]])
    first_elem = first(test_arr)
    assert np.isclose(first_elem, first_elem_ref, equal_nan=True).all()
    test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
    first_elem_ref = np.array([[[np.nan, 2.], [1., 2.]]])
    first_elem = first(test_arr, ignore_nodata=False)
    assert np.isclose(first_elem, first_elem_ref, equal_nan=True).all()

def test_last():
    assert last([1, 0, 3, 2]) == 2
    assert np.isnan(last([0, 1, np.nan], ignore_nodata=False))
    assert np.isnan(last([]))

    test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 3]], [[1, 2], [1, np.nan]]])
    last_elem_ref = np.array([[[1., 2.], [1., 3.]]])
    last_elem = last(test_arr)
    assert np.isclose(last_elem, last_elem_ref, equal_nan=True).all()
    test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
    last_elem_ref = np.array([[[1., 2.], [1., np.nan]]])
    last_elem = last(test_arr, ignore_nodata=False)
    assert np.isclose(last_elem, last_elem_ref, equal_nan=True).all()

def test_order():
    assert (order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9])==np.array([1, 2, 8, 5, 0, 4, 7, 9, 10, 3, 6])).all()
    assert (order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], nodata=True)==[1, 2, 8, 5, 0, 4, 7, 9, 10, 3, 6]).all()
    assert (order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=True)==[9, 10, 7, 4, 0, 5, 8, 2, 1, 3, 6]).all()
    assert (order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=False)==[6, 3, 9, 10, 7, 4, 0, 5, 8, 2, 1]).all()

def test_rearrange():
    assert (rearrange([5, 4, 3], [2, 1, 0]) == [3, 4, 5]).all()
    assert (rearrange([5, 4, 3, 2], [0, 2, 1, 3]) == [5, 3, 4, 2]).all()
    assert (rearrange([5, 4, 3, 2], [1, 3]) == [4, 2]).all()

def test_sort():
    """ Tests `sort` function. """
    assert (sort([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9])==[-1, 2, 3, 4, 6, 7, 8, 9, 9]).all()
    assert np.isclose(sort([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=True),
                        [9, 9, 8, 7, 6, 4, 3, 2, -1, np.nan, np.nan], equal_nan=True).all()
