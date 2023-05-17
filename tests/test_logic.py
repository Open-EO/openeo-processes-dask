from functools import partial

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.cubes import *
from openeo_processes_dask.process_implementations.logic import *
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


def test_and_():
    assert and_(True, True)
    assert not and_(True, False)
    assert not and_(False, False)
    assert not and_(False, np.nan)
    assert np.isnan(and_(True, np.nan))
    assert np.isclose(
        and_(
            x=[True, True, False, False, True], y=[True, False, False, np.nan, np.nan]
        ),
        [True, False, False, False, np.nan],
        equal_nan=True,
    ).all()


def test_or_():
    assert or_(True, True)
    assert or_(True, False)
    assert not or_(False, False)
    assert or_(True, np.nan)
    assert or_(np.nan, True)
    assert np.isnan(or_(False, np.nan))
    assert np.isclose(
        or_(x=[True, True, False, False, True], y=[True, False, False, np.nan, np.nan]),
        [True, True, False, np.nan, True],
        equal_nan=True,
    ).all()


def test_xor():
    assert not xor(True, True)
    assert not xor(False, False)
    assert xor(True, False)
    assert np.isnan(xor(True, np.nan))
    assert np.isnan(xor(False, np.nan))
    assert np.isclose(
        xor(x=[True, True, False, False, True], y=[True, False, False, np.nan, np.nan]),
        [False, True, False, np.nan, np.nan],
        equal_nan=True,
    ).all()


def test_not():
    assert not not_(True)
    assert not_(False)
    assert np.isnan(not_(np.nan))
    assert np.isclose(
        not_(x=[True, False, np.nan]),
        [False, True, np.nan],
        equal_nan=True,
    ).all()


def test_if():
    assert if_(True, "A", "B") == "A"
    assert if_(None, "A", "B") == "B"
    assert (if_(False, [1, 2, 3], [4, 5, 6]) == [4, 5, 6]).all()
    assert if_(True, 123) == 123
    assert np.isnan(if_(False, 1))
    assert np.isclose(
        if_(value=[True, None, False], accept=[1, 2, 3], reject=[4, 5, 6]),
        [1, 5, 6],
    ).all()


def test_any():
    assert not any_([False, np.nan])
    assert any_([True, np.nan])
    assert np.isnan(any_([False, np.nan], ignore_nodata=False))
    assert any_([True, np.nan], ignore_nodata=False)
    assert any_([True, False, True, False])
    assert any_([True, False])
    assert not any_([False, False])
    assert any_([True])
    assert np.isnan(any_([np.nan], ignore_nodata=False))
    assert np.isnan(any_([]))
    assert np.isclose(
        any_([[True, np.nan], [False, False]], ignore_nodata=False, axis=0),
        [True, np.nan],
        equal_nan=True,
    ).all()


def test_all():
    assert not all_([False, np.nan])
    assert all_([True, np.nan])
    assert not all_([False, np.nan], ignore_nodata=False)
    assert np.isnan(all_([True, np.nan], ignore_nodata=False))
    assert not all_([True, False, True, False])
    assert not all_([True, False])
    assert all_([True, True])
    assert all_([True])
    assert np.isnan(all_([np.nan], ignore_nodata=False))
    assert np.isnan(all_([]))
    assert np.isclose(
        all_([[True, np.nan], [False, True]], ignore_nodata=False, axis=0),
        [False, np.nan],
        equal_nan=True,
    ).all()


@pytest.mark.parametrize("size", [(30, 30, 20, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_reduce_dimension(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )

    input_cube[
        :, :, :, 0
    ] = True  # set all values in the first band to True - any() over bands will return True (ones_like)
    _process = partial(
        process_registry["any"].implementation,
        ignore_nodata=False,
        data=ParameterReference(from_parameter="data"),
    )
    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="bands")
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert output_cube.dims == ("x", "y", "t")
    assert isinstance(output_cube.data, da.Array)
    xr.testing.assert_equal(output_cube, xr.ones_like(output_cube))

    input_cube[
        :, :, :, 1
    ] = False  # set all values in the second band to False - all() over bands will return False (zeros_like)
    _process = partial(
        process_registry["all"].implementation,
        ignore_nodata=False,
        data=ParameterReference(from_parameter="data"),
    )
    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="bands")
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )
    assert output_cube.dims == ("x", "y", "t")
    assert isinstance(output_cube.data, da.Array)
    xr.testing.assert_equal(output_cube, xr.zeros_like(output_cube))


@pytest.mark.parametrize("size", [(6, 5, 4, 2)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_merge_cubes(
    temporal_interval, bounding_box, random_raster_data, process_registry
):
    origin_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B01", "B02"],
        backend="dask",
    )

    cube_1 = origin_cube.sel({"bands": "B01"})
    cube_2 = origin_cube.sel({"bands": "B02"})
    cube_1[:, :, :] = True
    cube_2[:, :, :] = False

    overlap_resolver = partial(
        process_registry["and"].implementation, x=cube_1, y=cube_2
    )
    merged_cube = merge_cubes(cube_1, cube_2, overlap_resolver=overlap_resolver)
    assert merged_cube.dims == ("x", "y", "t")
    assert isinstance(merged_cube.data, da.Array)
    xr.testing.assert_equal(
        merged_cube, xr.zeros_like(merged_cube)
    )  # and(True, False) == False (zeros_like)

    overlap_resolver = partial(
        process_registry["or"].implementation, x=cube_1, y=cube_2
    )
    merged_cube = merge_cubes(cube_1, cube_2, overlap_resolver=overlap_resolver)
    assert merged_cube.dims == ("x", "y", "t")
    assert isinstance(merged_cube.data, da.Array)
    xr.testing.assert_equal(
        merged_cube, xr.ones_like(merged_cube)
    )  # or(True, False) == True (ones_like)

    overlap_resolver = partial(
        process_registry["xor"].implementation, x=cube_1, y=cube_2
    )
    merged_cube = merge_cubes(cube_1, cube_2, overlap_resolver=overlap_resolver)
    assert merged_cube.dims == ("x", "y", "t")
    assert isinstance(merged_cube.data, da.Array)
    xr.testing.assert_equal(
        merged_cube, xr.ones_like(merged_cube)
    )  # xor(True, False) == True (ones_like)


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
        backend="dask",
    )
    input_cube[:, :, :, :2] = True
    input_cube[:, :, :, 2:] = False

    _process = partial(
        process_registry["not"].implementation, x=ParameterReference(from_parameter="x")
    )
    output_cube = apply(data=input_cube, process=_process)
    expected_result = xr.zeros_like(input_cube)
    expected_result[:, :, :, 2:] = True
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=(expected_result),
    )
    assert isinstance(output_cube.data, da.Array)
    xr.testing.assert_equal(output_cube, expected_result)

    _process = partial(
        process_registry["if"].implementation,
        value=ParameterReference(from_parameter="x"),
        accept=True,
        reject=True,
    )
    output_cube = apply(data=input_cube, process=_process)
    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
        expected_results=(xr.ones_like(input_cube)),
    )
    assert isinstance(output_cube.data, da.Array)
    xr.testing.assert_equal(output_cube, xr.ones_like(input_cube))
