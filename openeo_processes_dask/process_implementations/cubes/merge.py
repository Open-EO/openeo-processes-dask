from openeo_processes_dask.process_implementations.data_model import RasterCube
from typing import Callable, Optional
import numpy as np
import xarray as xr
import geopandas as gpd
import dask_geopandas

__all__ = ["merge_cubes"]


def merge_cubes(
    cube1: RasterCube, cube2: RasterCube, overlap_resolver: Callable = None, context: Optional[dict] = None, **kwargs
):
    if context is None:
        context = {}
    if not isinstance(cube1, type(cube2)):
        raise Exception(f"Provided cubes have incompatible types. cube1: {type(cube1)}, cube2: {type(cube2)}")
    
    is_geopandas = isinstance(cube1, gpd.geodataframe.GeoDataFrame) and isinstance(cube2, gpd.geodataframe.GeoDataFrame)
    is_delayed_geopandas = isinstance(cube1, dask_geopandas.core.GeoDataFrame) and isinstance(cube2, dask_geopandas.core.GeoDataFrame)
    if is_geopandas or is_delayed_geopandas:
        if list(cube1.columns) == list(cube2.columns):
            if is_delayed_geopandas:
                merged_cube = cube1.append(cube2)
            if is_geopandas:
                merged_cube = cube1.append(cube2, ignore_index=True)
            print("Warning - Overlap resolver is not implemented for geopandas vector-cubes, cubes are simply appended!")
        else:
            if 'geometry' in cube1.columns and 'geometry' in cube2.columns:
                merged_cube = cube1.merge(cube2, on='geometry')
        return merged_cube

    if (cube1.dims == cube2.dims):  # Check if the dimensions have the same names
        matching = 0
        not_matching = 0
        relevant_coords = [c for c in cube1.coords if c != "spatial_ref"]

        for c in relevant_coords:
            coords_match = np.array_equal(cube1[c].values, cube2[c].values)

            if coords_match:  # dimension with different coordinates
                dimension = c
                matching += 1
            else:
                not_matching += 1
                dim_not_matching = c  # dimensions with some matching coordinates
        if matching == len(relevant_coords):  # all dimensions match
            if overlap_resolver is None:  # no overlap resolver, so a new dimension is added
                merge = xr.concat([cube1, cube2], dim='cubes')
                merge['cubes'] = ["Cube01", "Cube02"]
            else:
                if callable(overlap_resolver):  # overlap resolver, for example add
                    merge = overlap_resolver(cube1, cube2, **context)
                elif isinstance(overlap_resolver, xr.core.dataarray.DataArray):
                    merge = overlap_resolver
                else:
                    raise Exception('OverlapResolverMissing')
        else:  # WIP
            if not_matching == 1:  # one dimension where some coordinates match, others do not, other dimensions match
                same1 = []
                diff1 = []
                index = 0
                for t in cube1[dim_not_matching]:  # count matching coordinates
                    if (t == cube2[dim_not_matching]).any():
                        same1.append(index)
                        index += 1
                    else:  # count different coordinates
                        diff1.append(index)
                        index += 1
                same2 = []
                diff2 = []
                index2 = 0
                for t in cube2[dim_not_matching]:
                    if (t == cube1[dim_not_matching]).any():
                        same2.append(index2)
                        index2 += 1
                    else:
                        diff2.append(index2)
                        index2 += 1
                if callable(overlap_resolver):
                    c1 = cube1.transpose(dim_not_matching, ...)
                    c2 = cube2.transpose(dim_not_matching, ...)
                    merge = overlap_resolver(c1[same1], c2[same2], **context)
                    if len(diff1) > 0:
                        values_cube1 = c1[diff1]
                        merge = xr.concat([merge, values_cube1], dim=dim_not_matching)
                    if len(diff2) > 0:
                        values_cube2 = c2[diff2]
                        merge = xr.concat([merge, values_cube2], dim=dim_not_matching)
                    merge = merge.sortby(dim_not_matching)
                    merge = merge.transpose(*cube1.dims)
                else:
                    merge = xr.concat([cube1, cube2], dim=dim_not_matching)
            else:
                merge = xr.concat([cube1, cube2], dim=dim_not_matching)
    else:  # if dims do not match - WIP
        if len(cube1.dims) < len(cube2.dims):
            c1 = cube1
            c2 = cube2
        else:
            c1 = cube2
            c2 = cube1
        check = []
        for c in c1.dims:
            check.append(c in c1.dims and c in c2.dims)
        for c in c2.dims:
            if not (c in c1.dims):
                dimension = c
        if np.array(check).all() and len(c2[dimension]) == 1 and callable(overlap_resolver):
            c2 = c2.transpose(dimension, ...)
            merge = overlap_resolver(c2[0], c1, **context)
        elif isinstance(overlap_resolver, xr.core.dataarray.DataArray):
            merge = overlap_resolver
        else:
            raise Exception('OverlapResolverMissing')
    for a in cube1.attrs:
        if a in cube2.attrs and (cube1.attrs[a] == cube2.attrs[a]):
            merge.attrs[a] = cube1.attrs[a]
    return merge
