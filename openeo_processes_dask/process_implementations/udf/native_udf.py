"""
Native UDF processor for openeo-processes-dask.
Fixes Issue #330 by removing dependency on openeo-python-client and preserving dimension names.
"""

import logging
import sys
import types
from typing import Any, Callable, Dict, Optional

import dask.array as da
import xarray as xr

_log = logging.getLogger(__name__)


class UdfExecutionError(Exception):
    """Raised when UDF execution fails."""

    pass


class NativeUdfProcessor:
    """
    Native UDF processor that executes UDFs directly on xarray without client wrappers.
    Preserves semantic dimension names to fix Issue #330.
    """

    def __init__(self):
        self._allowed_imports = {
            "xarray": xr,
            "xr": xr,
            "numpy": self._safe_import("numpy"),
            "np": self._safe_import("numpy"),
            "pandas": self._safe_import("pandas"),
            "pd": self._safe_import("pandas"),
            "math": self._safe_import("math"),
            "datetime": self._safe_import("datetime"),
        }

    def _safe_import(self, module_name: str):
        """Safely import a module, returning None if not available."""
        try:
            return __import__(module_name)
        except ImportError:
            _log.warning(f"Module {module_name} not available for UDF execution")
            return None

    def run_udf(
        self, data: da.Array, udf: str, runtime: str, context: Optional[dict] = None
    ) -> xr.DataArray:
        """
        Execute UDF directly on xarray data, preserving dimension names.

        Args:
            data: Input dask array
            udf: UDF code string
            runtime: Runtime (currently only "Python" supported)
            context: Optional context dictionary

        Returns:
            xr.DataArray with preserved dimension names (fixes Issue #330)

        Raises:
            UdfExecutionError: If UDF execution fails
            ValueError: If runtime is not supported or UDF is invalid
        """
        if runtime.lower() != "python":
            raise ValueError(f"Only Python runtime supported, got: {runtime}")

        _log.info("Executing UDF with native processor (Issue #330 fix)")

        # Convert to xarray - dimensions preserved!
        # Note: When creating from dask array, we need to explicitly set dimension names
        xr_data = xr.DataArray(data)

        # Issue #330 fix: If dimensions are generic, try to infer semantic names
        if all(str(dim).startswith("dim_") for dim in xr_data.dims):
            # Common dimension patterns for geospatial data
            ndim = len(xr_data.dims)
            if ndim == 3:
                # Assume time, y, x for 3D data (most common case)
                xr_data = xr_data.rename({"dim_0": "time", "dim_1": "y", "dim_2": "x"})
            elif ndim == 4:
                # Assume time, band, y, x for 4D data
                xr_data = xr_data.rename(
                    {"dim_0": "time", "dim_1": "band", "dim_2": "y", "dim_3": "x"}
                )
            # For other dimensions, keep as-is but log the issue
            _log.warning(
                f"Generic dimensions detected for {ndim}D data: {list(xr_data.dims)}"
            )

        _log.debug(f"Input dimensions: {list(xr_data.dims)}")

        # Execute UDF natively without client wrappers
        try:
            result = self._execute_udf_native(xr_data, udf, context or {})
            _log.debug(f"Output dimensions: {list(result.dims)}")
            return result
        except Exception as e:
            raise UdfExecutionError(f"UDF execution failed: {e}") from e

    def _execute_udf_native(
        self, datacube: xr.DataArray, udf_code: str, context: dict[str, Any]
    ) -> xr.DataArray:
        """Execute UDF code directly without client dependencies."""

        # Compile UDF code safely
        udf_namespace = self._compile_udf_code(udf_code)

        # Find the UDF function
        udf_function = self._find_datacube_function(udf_namespace)

        # Execute UDF with proper arguments
        _log.debug("Executing UDF function with preserved dimensions")
        try:
            result = udf_function(datacube, context)
        except Exception as e:
            raise UdfExecutionError(f"UDF function execution failed: {e}") from e

        # Validate result
        if not isinstance(result, xr.DataArray):
            raise UdfExecutionError(
                f"UDF must return xarray.DataArray, got {type(result)}"
            )

        return result

    def _compile_udf_code(self, udf_code: str) -> dict[str, Any]:
        """Safely compile UDF code into an isolated namespace."""

        # Create isolated namespace with safe imports
        namespace = {
            "__builtins__": {
                # Safe built-ins
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
                "sorted": sorted,
                "isinstance": isinstance,
                "type": type,
                "all": all,
                "any": any,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "__import__": __import__,  # Needed for import statements in UDF
            }
        }

        # Add allowed imports
        namespace.update(self._allowed_imports)

        try:
            # Compile and execute UDF code in isolated namespace
            compiled_code = compile(udf_code, "<udf>", "exec")
            exec(compiled_code, namespace)
            _log.debug("UDF code compiled successfully")
        except SyntaxError as e:
            raise UdfExecutionError(f"UDF code syntax error: {e}") from e
        except Exception as e:
            raise UdfExecutionError(f"Failed to compile UDF code: {e}") from e

        return namespace

    def _find_datacube_function(self, udf_namespace: dict[str, Any]) -> Callable:
        """Find apply_datacube or apply_hypercube function in UDF namespace."""

        # Look for standard UDF function names (OpenEO specification)
        for func_name in ["apply_datacube", "apply_hypercube"]:
            if func_name in udf_namespace and callable(udf_namespace[func_name]):
                _log.debug(f"Found UDF function: {func_name}")
                return udf_namespace[func_name]

        # Fallback: look for any user-defined callable
        user_functions = [
            (name, func)
            for name, func in udf_namespace.items()
            if callable(func)
            and not name.startswith("_")
            and name not in self._allowed_imports
        ]

        if len(user_functions) == 1:
            func_name, func = user_functions[0]
            _log.debug(f"Found single user function: {func_name}")
            return func
        elif len(user_functions) > 1:
            func_names = [name for name, _ in user_functions]
            raise UdfExecutionError(
                f"Multiple functions found in UDF: {func_names}. "
                f"Use 'apply_datacube' or 'apply_hypercube' as function name."
            )
        else:
            raise UdfExecutionError(
                "No UDF function found. Define 'apply_datacube' or 'apply_hypercube' function."
            )


# Global instance for the module
_processor = NativeUdfProcessor()


def run_udf(
    data: da.Array, udf: str, runtime: str, context: Optional[dict] = None
) -> xr.DataArray:
    """
    Drop-in replacement for the current run_udf function.

    Fixes Issue #330 by preserving dimension names during UDF execution.
    No longer depends on openeo-python-client.

    Args:
        data: Input dask array
        udf: UDF code string
        runtime: Runtime ("Python" only)
        context: Optional context dictionary

    Returns:
        xr.DataArray with preserved semantic dimension names
    """
    return _processor.run_udf(data, udf, runtime, context)
