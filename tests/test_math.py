import dask.array as da
import numpy as np
import pytest

from openeo_processes_dask.process_implementations.exceptions import MinMaxSwapped
from openeo_processes_dask.process_implementations.math import *


def test_quantiles():
    quantiles_1 = quantiles(
        data=np.array([2, 4, 4, 4, 5, 5, 7, 9]),
        probabilities=[0.005, 0.01, 0.02, 0.05, 0.1, 0.5],
    )
    quantiles_1 = [_round(quantile, p=2) for quantile in quantiles_1]
    assert quantiles_1 == [2.07, 2.14, 2.28, 2.7, 3.4, 4.5]
    quantiles_2 = quantiles(data=np.array([2, 4, 4, 4, 5, 5, 7, 9]), probabilities=4)
    quantiles_2 = [_round(quantile, p=2) for quantile in quantiles_2]
    assert quantiles_2 == [4, 4.5, 5.5]
    quantiles_3 = quantiles(data=np.array([-1, -0.5, np.nan, 1]), probabilities=[2])
    quantiles_3 = [_round(quantile, p=2) for quantile in quantiles_3]
    assert quantiles_3 == [-0.5]
    quantiles_4 = quantiles(
        data=np.array([-1, -0.5, np.nan, 1]), q=4, ignore_nodata=False
    )
    assert (
        np.all([np.isnan(quantile) for quantile in quantiles_4])
        and len(quantiles_4) == 3
    )
    quantiles_5 = quantiles(data=np.array([]), probabilities=[0.1, 0.5])
    assert (
        np.all([np.isnan(quantile) for quantile in quantiles_5])
        and len(quantiles_5) == 2
    )


def test_sum():
    assert _sum([5, 1]) == 6
    assert _sum([-2, 4, 2.5]) == 4.5
    assert np.isnan(_sum([1, np.nan], ignore_nodata=False))


def test_e():
    assert e() == pytest.approx(2.71828182846)


def test_pi():
    assert pi() == pytest.approx(3.14159265359)


@pytest.mark.parametrize(
    "array,expected,ignore_nodata,axis",
    [
        (np.array([5, 0]), 0, True, None),
        (np.array([-2, 4, 2.5]), -20, True, None),
        (np.array([1, np.nan]), "nan", False, None),
        (np.array([-1]), -1, True, None),
        (np.array([np.nan]), "nan", False, None),
        (np.array([]), "nan", True, None),
        (np.array([[0, 2, 4], [5, 4, 3]]), np.array([0, 8, 12]), True, 0),
        (np.array([[0, 2, 4], [5, 4, 3]]), np.array([0, 60]), True, 1),
    ],
)
def test_product(array, expected, ignore_nodata, axis):
    array = np.array(array)
    result_np = product(array, ignore_nodata=ignore_nodata, axis=axis)
    if isinstance(expected, str) and "nan" == expected:
        assert np.isnan(result_np)
    else:
        assert np.array_equal(result_np, expected, equal_nan=True)

    result_dask = product(
        da.from_array(array), ignore_nodata=ignore_nodata, axis=axis
    ).compute()

    assert np.array_equal(result_dask, result_np, equal_nan=True)


@pytest.mark.parametrize(
    "x,y,expected",
    [(5, 3, 0.25), (1, 1, 0), (np.array([1, 1]), np.array([0, 1]), np.array([1, 0]))],
)
def test_normalized_difference(x, y, expected):
    result_np = normalized_difference(x, y)
    assert np.array_equal(result_np, expected, equal_nan=True)

    result_dask = normalized_difference(da.from_array(x), da.from_array(y))
    assert np.array_equal(result_np, result_dask.compute(), equal_nan=True)


def test_linear_scale_range():
    array = np.array([5, 0])

    dask_array = da.from_array(array)
    result_dask = linear_scale_range(dask_array, inputMin=0, inputMax=5)
    assert np.array_equal(np.array([1, 0]), result_dask.compute(), equal_nan=True)
    result_dask = linear_scale_range(dask_array, inputMin=5, inputMax=0)
    assert np.array_equal(np.array([0, 1]), result_dask.compute(), equal_nan=True)


def test_clip():
    array = np.array([5, 0])
    result_np = clip(array, min=0, max=2)

    dask_array = da.from_array(array)
    result_dask = clip(dask_array, min=0, max=2)
    assert np.array_equal(result_np, result_dask.compute(), equal_nan=True)
    assert np.array_equal(result_np, np.array([2, 0]))
    with pytest.raises(MinMaxSwapped):
        clip(dask_array, 10, 0)


def test_extrema():
    array_list = [0, 5, 10]
    result_np = np.array([0, 10])

    result = extrema(array_list)
    assert np.array_equal(result_np, result.compute())
    dask_array = da.from_array(np.array(array_list))
    result = extrema(dask_array, ignore_nodata=True, axis=0, keepdims=False)
    assert np.array_equal(result_np, result.compute())
