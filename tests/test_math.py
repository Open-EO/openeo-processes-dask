import numpy as np

from openeo_processes_dask.process_implementations.math import *


def test_quantiles():
    quantiles_1 = quantiles(
        data=np.array([2, 4, 4, 4, 5, 5, 7, 9]),
        probabilities=[0.005, 0.01, 0.02, 0.05, 0.1, 0.5],
    )
    quantiles_1 = [_round(quantile, p=2) for quantile in quantiles_1]
    assert quantiles_1 == [2.07, 2.14, 2.28, 2.7, 3.4, 4.5]
    quantiles_2 = quantiles(data=np.array([2, 4, 4, 4, 5, 5, 7, 9]), q=4)
    quantiles_2 = [_round(quantile, p=2) for quantile in quantiles_2]
    assert quantiles_2 == [4, 4.5, 5.5]
    quantiles_3 = quantiles(data=np.array([-1, -0.5, np.nan, 1]), q=2)
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
