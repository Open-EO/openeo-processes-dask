import numpy as np
import pytest
import xarray as xr

import openeo_processes_dask.process_implementations.cubes.indices as indices
from openeo_processes_dask.process_implementations.cubes.indices import ndvi
from openeo_processes_dask.process_implementations.exceptions import (
    BandExists,
    DimensionAmbiguous,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(20, 20, 10, 2)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_ndvi(temporal_interval, bounding_box, random_raster_data, process_registry):
    input_cube = create_fake_rastercube(
        data=random_raster_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["red", "nir"],
        backend="dask",
    )

    # Test whether this works with different band names
    input_cube = input_cube.rename("s2")
    input_cube = input_cube.rename({"bands": "b"})
    input_cube = input_cube.assign_coords(common_name=("b", ["red", "nir"]))

    output = ndvi(input_cube)

    band_dim = input_cube.openeo.band_dims[0]
    assert band_dim not in output.dims

    expected_results = (
        input_cube.sel({band_dim: "nir"}) - input_cube.sel({band_dim: "red"})
    ) / (input_cube.sel({band_dim: "nir"}) + input_cube.sel({band_dim: "red"}))

    general_output_checks(
        input_cube=input_cube, output_cube=output, expected_results=expected_results
    )

    cube_with_resolvable_coords = input_cube.assign_coords(
        {band_dim: ["blue", "yellow"]}
    )
    output = ndvi(cube_with_resolvable_coords)
    general_output_checks(
        input_cube=cube_with_resolvable_coords,
        output_cube=output,
        expected_results=expected_results,
    )

    with pytest.raises(DimensionAmbiguous):
        ndvi(output)

    cube_with_nir_unresolvable = cube_with_resolvable_coords
    cube_with_nir_unresolvable.common_name.data = np.array(["blue", "red"])

    with pytest.raises(NirBandAmbiguous):
        ndvi(cube_with_nir_unresolvable)

    cube_with_red_unresolvable = cube_with_resolvable_coords
    cube_with_red_unresolvable.common_name.data = np.array(["nir", "yellow"])

    with pytest.raises(RedBandAmbiguous):
        ndvi(cube_with_red_unresolvable)

    cube_with_nothing_resolvable = cube_with_resolvable_coords
    cube_with_nothing_resolvable = cube_with_nothing_resolvable.drop_vars("common_name")
    with pytest.raises(KeyError):
        ndvi(cube_with_nothing_resolvable)

    target_band = "yayyyy"
    output_with_extra_dim = ndvi(input_cube, target_band=target_band)
    assert isinstance(output_with_extra_dim, xr.DataArray)
    assert len(output_with_extra_dim.dims) == len(output.dims) + 1
    assert (
        len(output_with_extra_dim.coords[band_dim])
        == len(input_cube.coords[band_dim]) + 1
    )
    assert output_with_extra_dim.coords[band_dim].values[-1] == target_band
    assert output_with_extra_dim.coords["common_name"].dtype == np.dtype("<U4")
    assert output_with_extra_dim.coords["common_name"][-1] == "NDVI"

    output_with_other_dim = ndvi(
        input_cube.rename({"common_name": "renamed"}), target_band=target_band
    )
    assert isinstance(output_with_other_dim, xr.DataArray)
    assert len(output_with_other_dim.dims) == len(output.dims) + 1
    assert (
        len(output_with_other_dim.coords[band_dim])
        == len(input_cube.coords[band_dim]) + 1
    )
    assert output_with_other_dim.coords[band_dim].values[-1] == target_band
    assert output_with_other_dim.coords["renamed"][-1] == ""

    # unnamed cube caused problems in the past
    input_cube_noname = input_cube.rename(None)
    out_noname = ndvi(input_cube_noname, target_band="ndvi")
    assert (
        len(out_noname.coords[band_dim]) == len(input_cube_noname.coords[band_dim]) + 1
    )

    with pytest.raises(BandExists):
        output_with_extra_dim = ndvi(input_cube, target_band="t")


def test_dummy_value():
    out = indices._dummy_value(np.float32)
    assert np.isnan(out)

    out = indices._dummy_value(np.datetime64("2026-09-01").dtype)
    assert np.isnat(out)

    out = indices._dummy_value(np.timedelta64(100).dtype)
    assert np.isnat(out)

    out = indices._dummy_value(np.int64)
    assert out == -1

    out = indices._dummy_value(np.dtype("bool"))
    assert out is False

    out = indices._dummy_value(np.dtype("<U5"))
    assert out == ""
