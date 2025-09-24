"""
Comprehensive test suite for native UDF implementation.
Tests Issue #330 fix and ensures no regressions.
"""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from openeo_processes_dask.process_implementations.udf.native_udf import (
    NativeUdfProcessor,
    UdfExecutionError,
)
from openeo_processes_dask.process_implementations.udf.udf import run_udf


class TestIssue330Fix:
    """Tests specifically for Issue #330: dimension name preservation."""

    def test_3d_semantic_dimensions_preserved(self):
        """Test that 3D data gets semantic dimension names (time, y, x)."""
        data = np.random.rand(3, 4, 5).astype(np.float32)
        dask_data = da.from_array(data)

        udf_code = """
def apply_datacube(cube, context):
    # Verify we get semantic dimensions, not dim_0, dim_1, dim_2
    dims = list(cube.dims)
    assert 'time' in dims, f"Expected 'time' in dimensions, got {dims}"
    assert 'y' in dims, f"Expected 'y' in dimensions, got {dims}"
    assert 'x' in dims, f"Expected 'x' in dimensions, got {dims}"

    # Verify no generic dimensions
    generic_dims = [d for d in dims if str(d).startswith('dim_')]
    assert not generic_dims, f"Found generic dimensions (Issue #330): {generic_dims}"

    return cube + 1
"""

        result = run_udf(dask_data, udf_code, "Python")

        # Verify result has semantic dimensions
        assert list(result.dims) == ["time", "y", "x"]
        assert result.shape == data.shape
        assert np.allclose(result.values, data + 1)

    def test_4d_semantic_dimensions_preserved(self):
        """Test that 4D data gets semantic dimension names (time, band, y, x)."""
        data = np.random.rand(3, 4, 5, 6).astype(np.float32)
        dask_data = da.from_array(data)

        udf_code = """
def apply_datacube(cube, context):
    dims = list(cube.dims)
    expected = ['time', 'band', 'y', 'x']
    assert dims == expected, f"Expected {expected}, got {dims}"

    # Verify no generic dimensions (Issue #330 fix)
    generic_dims = [d for d in dims if str(d).startswith('dim_')]
    assert not generic_dims, f"Issue #330 not fixed: {generic_dims}"

    return cube * 2
"""

        result = run_udf(dask_data, udf_code, "Python")

        assert list(result.dims) == ["time", "band", "y", "x"]
        assert result.shape == data.shape
        assert np.allclose(result.values, data * 2)

    def test_dimension_preservation_consistency(self):
        """Test that dimensions are consistently preserved across UDF operations."""
        data = np.random.rand(2, 3, 4).astype(np.float32)
        dask_data = da.from_array(data)

        udf_code = """
import xarray as xr

def apply_datacube(cube, context):
    # Test various xarray operations preserve dimensions
    result = cube + 1
    result = result * 2
    result = result.mean('time')  # Should preserve y, x dimensions

    # Add time dimension back for consistency
    result = result.expand_dims('time')

    return result
"""

        result = run_udf(dask_data, udf_code, "Python")

        # Should have semantic dimensions
        expected_dims = ["time", "y", "x"]
        assert (
            list(result.dims) == expected_dims
        ), f"Expected {expected_dims}, got {list(result.dims)}"


class TestNativeUdfProcessor:
    """Tests for the NativeUdfProcessor class."""

    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        processor = NativeUdfProcessor()
        assert processor is not None
        assert hasattr(processor, "run_udf")

    def test_python_runtime_only(self):
        """Test that only Python runtime is supported."""
        processor = NativeUdfProcessor()
        data = da.from_array(np.random.rand(2, 3))

        with pytest.raises(ValueError, match="Only Python runtime supported"):
            processor.run_udf(data, "return data", "R")

        with pytest.raises(ValueError, match="Only Python runtime supported"):
            processor.run_udf(data, "return data", "JavaScript")

    def test_udf_function_discovery(self):
        """Test UDF function discovery works for different patterns."""
        processor = NativeUdfProcessor()
        data = da.from_array(np.random.rand(2, 3))

        # Test apply_datacube
        udf_datacube = """
def apply_datacube(cube, context):
    return cube + 1
"""
        result = processor.run_udf(data, udf_datacube, "Python")
        assert result is not None

        # Test apply_hypercube
        udf_hypercube = """
def apply_hypercube(cube, context):
    return cube * 2
"""
        result = processor.run_udf(data, udf_hypercube, "Python")
        assert result is not None

    def test_udf_error_handling(self):
        """Test UDF error handling and validation."""
        processor = NativeUdfProcessor()
        data = da.from_array(np.random.rand(2, 3))

        # Test syntax error
        with pytest.raises(UdfExecutionError, match="UDF code syntax error"):
            processor.run_udf(data, "invalid python syntax !!!", "Python")

        # Test no function found
        with pytest.raises(UdfExecutionError, match="No UDF function found"):
            processor.run_udf(data, "x = 5", "Python")

        # Test multiple functions
        udf_multiple = """
def func1(cube, context):
    return cube
def func2(cube, context):
    return cube
"""
        with pytest.raises(UdfExecutionError, match="Multiple functions found"):
            processor.run_udf(data, udf_multiple, "Python")

        # Test invalid return type
        udf_invalid_return = """
def apply_datacube(cube, context):
    return "not an xarray"
"""
        with pytest.raises(UdfExecutionError, match="must return xarray.DataArray"):
            processor.run_udf(data, udf_invalid_return, "Python")


class TestBackwardCompatibility:
    """Tests to ensure existing UDF patterns still work."""

    def test_existing_test_pattern(self):
        """Test that existing test patterns from test_udf.py still work."""
        data = np.random.rand(6, 5, 4, 4).astype(np.float32)
        dask_data = da.from_array(data)

        # Original test UDF pattern
        udf = """
import xarray as xr
def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    return cube + 1
"""

        result = run_udf(data=dask_data, udf=udf, runtime="Python")

        # Should work and have semantic dimensions
        assert result.shape == data.shape
        assert np.allclose(result.values, data + 1)
        # 4D data should get time, band, y, x
        assert list(result.dims) == ["time", "band", "y", "x"]

    def test_context_parameter_passing(self):
        """Test that context parameter is properly passed to UDF."""
        data = da.from_array(np.random.rand(2, 3, 4))
        context = {"multiplier": 3, "offset": 10}

        udf_code = """
def apply_datacube(cube, context):
    multiplier = context.get("multiplier", 1)
    offset = context.get("offset", 0)
    return cube * multiplier + offset
"""

        result = run_udf(data, udf_code, "Python", context)
        expected = data.compute() * 3 + 10
        assert np.allclose(result.values, expected)

    def test_xarray_operations_available(self):
        """Test that xarray operations are available in UDF."""
        data = da.from_array(np.random.rand(3, 4, 5))

        udf_code = """
import xarray as xr
import numpy as np

def apply_datacube(cube, context):
    # Test various xarray operations
    result = cube.mean('time')
    result = result.expand_dims('time')
    result = xr.concat([result, result], dim='time')
    return result
"""

        result = run_udf(data, udf_code, "Python")
        assert result.shape[0] == 2  # Should have 2 time steps
        assert result.shape[1:] == data.shape[1:]  # Other dims preserved


class TestIssue330Reproduction:
    """Tests that reproduce Issue #330 scenarios from our original scripts."""

    def test_issue_330_reproduction_script(self):
        """Reproduce the exact scenario from our Issue #330 detection script."""
        # Simulate the same data pattern from our reproduction scripts
        data = np.random.rand(3, 4, 5)  # time, y, x
        dask_data = da.from_array(data)

        # UDF that detects Issue #330
        udf_code = """
def apply_datacube(cube, context):
    dims = list(cube.dims)
    semantic = [d for d in dims if d in ['x', 'y', 'time']]
    generic = [d for d in dims if d.startswith('dim_')]

    print(f"Dims: {dims}")
    if semantic:
        print(f"✅ Semantic: {semantic}")
    if generic:
        print(f"❌ Issue #330: {generic}")

    # The key test: should have semantic dimensions
    assert semantic, f"No semantic dimensions found: {dims}"
    assert not generic, f"Generic dimensions found (Issue #330): {generic}"

    return cube + 1
"""

        result = run_udf(dask_data, udf_code, "Python")

        # Verify the fix
        assert "time" in result.dims
        assert "y" in result.dims
        assert "x" in result.dims
        assert not any(d.startswith("dim_") for d in result.dims)


if __name__ == "__main__":
    # Run basic tests if executed directly
    test_suite = TestIssue330Fix()
    test_suite.test_3d_semantic_dimensions_preserved()
    test_suite.test_4d_semantic_dimensions_preserved()
    print("✅ All Issue #330 tests passed!")
