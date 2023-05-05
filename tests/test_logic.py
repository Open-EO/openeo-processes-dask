import dask.array as da
import numpy as np
import pytest

from openeo_processes_dask.process_implementations.logic import is_valid


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
