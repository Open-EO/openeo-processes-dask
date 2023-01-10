from typing import Callable, Optional

import numpy as np
import xarray as xr

from openeo_processes_dask.exceptions import OverlapResolverMissing
from openeo_processes_dask.process_implementations.data_model import RasterCube

__all__ = ["merge_cubes"]

NEW_DIM_NAME = "cubes"


from collections import namedtuple

Overlap = namedtuple("Overlap", ["only_in_cube1", "only_in_cube2", "in_both"])


def merge_cubes(
    cube1: RasterCube,
    cube2: RasterCube,
    overlap_resolver: Callable = None,
    context: Optional[dict] = None,
    **kwargs,
) -> RasterCube:

    if context is None:
        context = {}
    if not isinstance(cube1, type(cube2)):
        raise Exception(
            f"Provided cubes have incompatible types. cube1: {type(cube1)}, cube2: {type(cube2)}"
        )

    # Key: dimension name
    # Value: (labels in cube1 not in cube2, labels in cube2 not in cube1)
    overlap_per_shared_dim = {
        dim: Overlap(
            only_in_cube1=np.setdiff1d(cube1[dim].data, cube2[dim].data),
            only_in_cube2=np.setdiff1d(cube2[dim].data, cube1[dim].data),
            in_both=np.intersect1d(cube1[dim].data, cube2[dim].data),
        )
        for dim in set(cube1.dims).intersection(set(cube2.dims))
    }

    if cube1.dims == cube2.dims:
        # Check whether all of the shared dims have exactly the same labels
        dims_have_no_label_diff = all(
            [
                len(overlap.only_in_cube1) == 0 and len(overlap.only_in_cube2) == 0
                for overlap in overlap_per_shared_dim.values()
            ]
        )
        if dims_have_no_label_diff:
            # Example 3: All dimensions and their labels are equal
            concat_both_cubes = xr.concat([cube1, cube2], dim=NEW_DIM_NAME)
            if overlap_resolver is None:
                # Example 3.1: Concat along new "cubes" dimension
                merged_cube = concat_both_cubes
            else:
                # Example 3.2: Elementwise operation
                merged_cube = concat_both_cubes.reduce(
                    overlap_resolver, dim=NEW_DIM_NAME, keep_attrs=True
                )
        else:
            # Example 1 & 2
            differing_dims = [
                dim
                for dim, overlap in overlap_per_shared_dim.items()
                if len(overlap.in_both) > 0
                and (len(overlap.only_in_cube1) > 0 or len(overlap.only_in_cube2) > 0)
            ]

            if len(differing_dims) == 0:
                # Example 1: No overlap on any dimensions, can just combine by coords
                merged_cube = xr.combine_by_coords([cube1, cube2])
            elif len(differing_dims) == 1:
                # Example 2: Overlap on one dimension, resolve these pixels with overlap resolver

                if overlap_resolver is None or not callable(overlap_resolver):
                    raise OverlapResolverMissing(
                        "Overlapping data cubes, but no overlap resolver has been specified."
                    )

                overlapping_dim = differing_dims[0]
                stacked_conflicts = xr.concat(
                    [
                        cube1.sel(
                            **{
                                overlapping_dim: overlap_per_shared_dim[
                                    overlapping_dim
                                ].in_both
                            }
                        ),
                        cube2.sel(
                            **{
                                overlapping_dim: overlap_per_shared_dim[
                                    overlapping_dim
                                ].in_both
                            }
                        ),
                    ],
                    dim="cubes",
                )
                merge_conflicts = stacked_conflicts.reduce(
                    overlap_resolver, dim="cubes"
                )

                rest_of_cube_1 = cube1.sel(
                    **{
                        overlapping_dim: overlap_per_shared_dim[
                            overlapping_dim
                        ].only_in_cube1
                    }
                )
                rest_of_cube_2 = cube2.sel(
                    **{
                        overlapping_dim: overlap_per_shared_dim[
                            overlapping_dim
                        ].only_in_cube2
                    }
                )
                merged_cube = xr.combine_by_coords(
                    [merge_conflicts, rest_of_cube_1, rest_of_cube_2]
                )

            else:
                raise ValueError(
                    "More than one overlapping dimension, merge not possible."
                )

    elif True:
        # Example 4: broadcast lower dimension cube to higher-dimension cube
        pass

    else:
        raise ValueError()

    return merged_cube

    if cube1.dims == cube2.dims:  # Check if the dimensions have the same names
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
            if (
                overlap_resolver is None
            ):  # Example 3.1 in https://processes.openeo.org/#merge_cubes: overlapping dims, but no overlap resolver, so a new dimension is added
                merge = xr.concat([cube1, cube2], dim="cubes")
                merge["cubes"] = ["Cube01", "Cube02"]
            else:
                if callable(
                    overlap_resolver
                ):  # Example 3.2 in https://processes.openeo.org/#merge_cubes: overlapping dims, overlap resolver for example add
                    parameters = {"x": cube1, "y": cube2, "context": context}
                    merge = overlap_resolver(parameters=parameters, dimension=dimension)
        else:  # WIP
            if (
                not_matching == 1
            ):  # one dimension where some coordinates match, others do not, other dimensions match
                if overlap_resolver is None or not callable(overlap_resolver):
                    raise OverlapResolverMissing(
                        "Overlapping data cubes, but no overlap resolver has been specified."
                    )
                if callable(
                    overlap_resolver
                ):  # Example 2 in https://processes.openeo.org/#merge_cubes: one dim not matching, overlap resolver for same coordinates
                    same1 = []
                    diff1 = []
                    for index, t in enumerate(
                        cube1[dim_not_matching]
                    ):  # count matching coordinates
                        if (t == cube2[dim_not_matching]).any():
                            same1.append(index)
                        else:  # count different coordinates
                            diff1.append(index)
                    same2 = []
                    diff2 = []
                    for index, t in enumerate(cube2[dim_not_matching]):
                        if (t == cube1[dim_not_matching]).any():
                            same2.append(index)
                        else:
                            diff2.append(index)
                    stacked_conflicts = xr.concat(
                        [
                            cube1.isel(**{dim_not_matching: same1}),
                            cube2.isel(**{dim_not_matching: same2}),
                        ],
                        dim="cubes",
                    )
                    merge = stacked_conflicts.reduce(overlap_resolver, dim="cubes")

                    if len(diff1) > 0:
                        values_cube1 = cube1.isel(**{dim_not_matching: diff1})
                        merge = xr.concat([merge, values_cube1], dim=dim_not_matching)
                    if len(diff2) > 0:
                        values_cube2 = cube2.isel(**{dim_not_matching: diff2})
                        merge = xr.concat([merge, values_cube2], dim=dim_not_matching)
                    merge = merge.sortby(dim_not_matching)

                else:  # Example 1 in https://processes.openeo.org/#merge_cubes: one dim not matching, no overlap resolver
                    merge = xr.concat([cube1, cube2], dim=dim_not_matching)
            else:
                merge = xr.concat([cube1, cube2], dim=dim_not_matching)
    else:  # Example 4 in https://processes.openeo.org/#merge_cubes: dim in one cube, but missing in other one, overlap resolver
        if len(cube1.dims) < len(cube2.dims):
            c1 = cube1
            c2 = cube2
        else:
            c1 = cube2
            c2 = cube1
        check = []
        for c in c1.dims:
            check.append(c in c2.dims)  # dims of smaller cube have to be in larger one
        if all(check) and callable(overlap_resolver):
            parameters = {"x": cube1, "y": cube2, "context": context}
            merge = overlap_resolver(parameters=parameters)
        else:
            raise OverlapResolverMissing(
                "Datacubes overlap, but no overlap resolver has been specified."
            )
    for a in cube1.attrs:
        if a in cube2.attrs and (cube1.attrs[a] == cube2.attrs[a]):
            merge.attrs[a] = cube1.attrs[a]
    return merge
