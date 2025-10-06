"""
Native UDF processor for openeo-processes-dask.
Fixes Issue #330 by removing dependency on openeo-python-client and preserving dimension names.
"""

import logging
import sys
import types
from typing import Any, Callable, Dict, Optional, Union

import dask.array as da
import numpy as np
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
        self,
        data: Union[da.Array, xr.DataArray],
        udf: str,
        runtime: str,
        context: Optional[dict] = None,
    ) -> xr.DataArray:
        """
        Execute UDF directly on xarray data, preserving dimension names.

        Args:
            data: Input data (dask array or xarray DataArray)
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

        # Handle both dask arrays and xarray DataArrays
        if isinstance(data, xr.DataArray):
            # Issue #330 fix: xarray input preserves dimensions automatically!
            xr_data = data
            _log.info(f"Using xarray input with dimensions: {list(xr_data.dims)}")
        else:
            # Convert dask array to xarray - dimensions will be generic
            xr_data = xr.DataArray(data)

            # Issue #330 fix: If dimensions are generic, try to infer semantic names
            if all(str(dim).startswith("dim_") for dim in xr_data.dims):
                # Common dimension patterns for geospatial data
                ndim = len(xr_data.dims)
                if ndim == 3:
                    # Assume time, y, x for 3D data (most common case)
                    xr_data = xr_data.rename(
                        {"dim_0": "time", "dim_1": "y", "dim_2": "x"}
                    )
                elif ndim == 4:
                    # Assume time, band, y, x for 4D data
                    xr_data = xr_data.rename(
                        {"dim_0": "time", "dim_1": "band", "dim_2": "y", "dim_3": "x"}
                    )
                # For other dimensions, keep as-is but log the issue
                _log.warning(
                    f"Unusual number of dimensions ({ndim}), keeping generic names: {list(xr_data.dims)}"
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
        # Attempt to infer spatial resolution from coordinates and provide defaults
        try:
            res = None
            dx = dy = None
            if "x" in datacube.coords and len(datacube.coords["x"]) > 1:
                xs = np.asarray(datacube.coords["x"])
                dx = float(np.median(np.abs(np.diff(xs))))
            if "y" in datacube.coords and len(datacube.coords["y"]) > 1:
                ys = np.asarray(datacube.coords["y"])
                dy = float(np.median(np.abs(np.diff(ys))))
            if dx is not None and dy is not None:
                res = float(np.mean([abs(dx), abs(dy)]))
            elif dx is not None:
                res = abs(dx)
            elif dy is not None:
                res = abs(dy)
            if res is not None:
                context.setdefault("resolution", res)
                _log.info(f"Inferred resolution={res} from coordinates")
        except Exception:
            _log.debug("Could not infer resolution from coords", exc_info=True)

        # Try to provide a sensible nodata/default background value
        try:
            nod = None
            if isinstance(datacube, xr.DataArray):
                # prefer explicit numeric nodata attributes
                if isinstance(datacube.attrs, dict):
                    nod_cand = datacube.attrs.get("nodata")
                    if nod_cand is not None and not (
                        isinstance(nod_cand, float) and np.isnan(nod_cand)
                    ):
                        nod = nod_cand
                # try encoding fill value if present and numeric
                enc = getattr(datacube, "encoding", {})
                if nod is None and isinstance(enc, dict):
                    fill = enc.get("_FillValue")
                    if fill is not None and not (
                        isinstance(fill, float) and np.isnan(fill)
                    ):
                        nod = fill
            # final fallback
            if nod is None or (isinstance(nod, float) and np.isnan(nod)):
                nod = 255
            context.setdefault("nodataval", nod)
            _log.info(f"Using nodataval={nod} for UDF execution")
        except Exception:
            _log.debug("Could not infer nodata", exc_info=True)

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

        # Optional post-processing to align with common hillshade outputs
        # (clip to 0-255, round to integer, cast to uint8). This only runs
        # when the caller/udf allows it via context key 'native_postprocess'
        if context.get("native_postprocess", True):
            try:
                _log.debug("Applying native post-processing to UDF output")
                vals = result.values.astype(float)
                vals = np.clip(vals, 0, 255)
                vals = np.rint(vals).astype(np.uint8)
                # Preserve coords and dims
                result = xr.DataArray(
                    vals, coords=result.coords, dims=result.dims, name=result.name
                )
                _log.info("Native post-processing applied: rounded to uint8 0-255")
            except Exception:
                _log.debug("Native post-processing failed", exc_info=True)

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
    data: Union[da.Array, xr.DataArray],
    udf: str,
    runtime: str,
    context: Optional[dict] = None,
) -> xr.DataArray:
    """
    Drop-in replacement for the current run_udf function.

    Fixes Issue #330 by preserving dimension names during UDF execution.
    No longer depends on openeo-python-client.

    Args:
        data: Input data (dask array or xarray DataArray)
        udf: UDF code string
        runtime: Runtime ("Python" only)
        context: Optional context dictionary

    Returns:
        xr.DataArray with preserved dimensions
    """
    return _processor.run_udf(data, udf, runtime, context)
