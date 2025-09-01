"""
UDF Dimension Helper

This module provides utilities to re-assign dimension names and labels 
at the beginning of UDFs to address issue #330 where UDFs receive 
generic dimension names (dim_0, dim_1, etc.) instead of semantic names.

Usage:
    from openeo_processes_dask.process_implementations.udf.dimension_helper import fix_udf_dimensions
    
    def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
        # Re-assign dimension names at the beginning
        cube = fix_udf_dimensions(cube, context)
        
        # Now you can use semantic names
        return cube.mean(dim='bands')  # instead of dim='dim_3'
"""

import xarray as xr
from typing import Optional, Dict, Any


def fix_udf_dimensions(
    cube: xr.DataArray, 
    context: Optional[Dict[str, Any]] = None
) -> xr.DataArray:
    """
    Re-assign generic dimension names (dim_0, dim_1, etc.) to semantic names using metadata.
    
    This function addresses issue #330 by dynamically extracting the original dimension names
    from the UDF context metadata and mapping them back to the generic dimension names.
    
    Args:
        cube: Input DataArray with potentially generic dimension names
        context: UDF context containing _openeo_dimension_metadata
        
    Returns:
        DataArray with semantic dimension names restored from metadata
        
    Example:
        >>> cube.dims  # ('dim_0', 'dim_1', 'dim_2', 'dim_3')
        >>> cube = fix_udf_dimensions(cube, context)
        >>> cube.dims  # ('time', 'y', 'x', 'bands')  # actual names from metadata
    """
    try:
        # Check if dimensions are already semantic
        if not any(str(dim).startswith('dim_') for dim in cube.dims):
            return cube
        
        # Extract dimension metadata from context
        if not context or '_openeo_dimension_metadata' not in context:
            # Fallback: if no metadata available, return as-is
            return cube
        
        metadata = context['_openeo_dimension_metadata']
        original_dimensions = metadata.get('all_dimensions', [])
        original_shape = metadata.get('data_shape', [])
        
        if not original_dimensions or not original_shape:
            return cube
        
        # Ensure we have the same number of dimensions
        if len(original_dimensions) != len(original_shape):
            return cube
            
        # Map current cube shape to original dimensions by matching sizes
        # The cube shape may be reordered due to apply_dimension operations
        semantic_dims = []
        used_dimensions = set()
        
        for current_size in cube.shape:
            # Find which original dimension has this size
            matching_dim = None
            for dim_name, orig_size in zip(original_dimensions, original_shape):
                if orig_size == current_size and dim_name not in used_dimensions:
                    matching_dim = dim_name
                    used_dimensions.add(dim_name)
                    break
            
            if matching_dim:
                semantic_dims.append(matching_dim)
            else:
                # If we can't match, use a safe fallback name
                fallback_name = f"dim_{len(semantic_dims)}"
                semantic_dims.append(fallback_name)
        
        # Ensure we have the correct number of dimension names
        if len(semantic_dims) != len(cube.dims):
            return cube
        
        # Create mapping from generic to semantic names
        dim_mapping = {}
        for i, (old_dim, new_dim) in enumerate(zip(cube.dims, semantic_dims)):
            # Only rename if it's actually a generic dimension name
            if str(old_dim).startswith('dim_'):
                dim_mapping[old_dim] = new_dim
        
        # Only proceed with renaming if we have valid mappings
        if dim_mapping:
            return cube.rename(dim_mapping)
        else:
            return cube
            
    except Exception as e:
        # If anything goes wrong, return the original cube
        # This ensures we never break the UDF execution
        import warnings
        warnings.warn(f"Dimension helper failed: {e}. Returning original cube.")
        return cube


def assign_dimension_labels(
    cube: xr.DataArray,
    context: Optional[Dict[str, Any]] = None,
    dimension_labels: Optional[Dict[str, Any]] = None
) -> xr.DataArray:
    """
    Assign coordinate labels to dimensions using metadata or provided labels.
    
    Args:
        cube: Input DataArray
        context: UDF context containing dimension coordinates in metadata
        dimension_labels: Optional manual override for coordinate values
                         Example: {'bands': ['B02', 'B03', 'B04'], 't': ['2022-01', '2022-02']}
        
    Returns:
        DataArray with assigned coordinate labels
    """
    try:
        new_coords = {}
        
        # First, try to get coordinates from metadata
        if context and '_openeo_dimension_metadata' in context:
            metadata = context['_openeo_dimension_metadata']
            dimension_coords = metadata.get('dimension_coords', {})
            
            for dim_name in cube.dims:
                if dim_name in dimension_coords:
                    coords = dimension_coords[dim_name]
                    if coords is not None and len(coords) == cube.sizes[dim_name]:
                        new_coords[dim_name] = coords
        
        # Override with manually provided labels if given
        if dimension_labels:
            for dim, labels in dimension_labels.items():
                if dim in cube.dims and labels is not None:
                    # Ensure labels match dimension size
                    if len(labels) == cube.sizes[dim]:
                        new_coords[dim] = labels
        
        # Only assign coordinates if we have valid ones
        if new_coords:
            return cube.assign_coords(new_coords)
        else:
            return cube
            
    except Exception as e:
        # If coordinate assignment fails, return the original cube
        import warnings
        warnings.warn(f"Coordinate assignment failed: {e}. Returning cube without coordinates.")
        return cube


def restore_semantic_dimensions(
    cube: xr.DataArray,
    context: Optional[Dict[str, Any]] = None,
    dimension_labels: Optional[Dict[str, Any]] = None
) -> xr.DataArray:
    """
    Complete solution: restore both semantic dimension names and labels from metadata.
    
    This is the main function to use at the beginning of UDFs. It dynamically extracts
    the original dimension information from the UDF context metadata.
    
    Args:
        cube: Input DataArray with generic dimension names
        context: UDF context containing _openeo_dimension_metadata
        dimension_labels: Optional manual override for coordinate labels
        
    Returns:
        DataArray with semantic dimensions and coordinate labels restored from metadata
        
    Example:
        def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
            # Complete fix at the beginning - uses metadata automatically
            cube = restore_semantic_dimensions(cube, context)
            
            # Now use actual semantic names from your data
            return cube.sel(bands='B02')  # Works with real band names
    """
    try:
        # First fix dimension names using metadata
        cube = fix_udf_dimensions(cube, context)
        
        # Then assign coordinate labels from metadata (with optional override)
        cube = assign_dimension_labels(cube, context, dimension_labels)
        
        return cube
        
    except Exception as e:
        # If anything fails, return the original cube to ensure UDF doesn't break
        import warnings
        warnings.warn(f"Semantic dimension restoration failed: {e}. Returning original cube.")
        return cube


# Main function alias for ease of use
def fix_dimensions(cube: xr.DataArray, **kwargs) -> xr.DataArray:
    """
    Convenience function - alias for restore_semantic_dimensions.
    
    This provides a short, easy-to-remember function name.
    """
    try:
        return restore_semantic_dimensions(cube, **kwargs)
    except Exception as e:
        # Ultimate fallback - if even the main function fails
        import warnings
        warnings.warn(f"Dimension fixing failed: {e}. Returning original cube.")
        return cube
