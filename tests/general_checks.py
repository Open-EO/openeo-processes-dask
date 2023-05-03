# Checks here are inspired by makepath/xarray-spatial/tests/general_checks.py
from typing import List

import dask.array as da
import numpy as np

from openeo_processes_dask.process_implementations.data_model import RasterCube


def general_output_checks(
    input_cube: RasterCube,
    output_cube: RasterCube,
    expected_results=None,
    verify_crs: bool = False,
    verify_attrs: bool = False,
    expected_dims: list = None,
    rtol=1e-06,
):
    assert isinstance(output_cube.data, type(input_cube.data))

    assert input_cube.openeo is not None
    assert output_cube.openeo is not None

    if verify_crs:
        assert input_cube.rio.crs == output_cube.rio.crs

    if verify_attrs:
        assert input_cube.attrs == output_cube.attrs

    if expected_results is not None:
        if isinstance(output_cube.data, np.ndarray):
            output_data = output_cube.data
        elif isinstance(output_cube.data, da.Array):
            output_data = output_cube.data.compute()
        else:
            raise TypeError(f"Unsupported array type: {type(output_cube.data)}")

        np.testing.assert_allclose(
            output_data, expected_results, equal_nan=True, rtol=rtol
        )

    if expected_dims is not None:
        actual_dims = output_cube.dims
        assert len(expected_dims) == len(actual_dims)
        assert set(actual_dims) == set(expected_dims)


def assert_numpy_equals_dask_numpy(numpy_cube, dask_cube, func):
    numpy_result = func(numpy_cube)
    dask_result = func(dask_cube)
    general_output_checks(dask_cube, dask_result)
    np.testing.assert_allclose(
        numpy_result.data, dask_result.data.compute(), equal_nan=True
    )
