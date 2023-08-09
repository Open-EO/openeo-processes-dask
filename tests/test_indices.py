import numpy as np
import pytest

from openeo_processes_dask.process_implementations.cubes.indices import ndvi
from openeo_processes_dask.process_implementations.cubes.load import load_stac
from openeo_processes_dask.process_implementations.exceptions import (
    BandExists,
    DimensionAmbiguous,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from tests.conftest import _random_raster_data
from tests.general_checks import general_output_checks


def test_ndvi(bounding_box):
    url = "./tests/data/stac/s2_l2a_test_item.json"
    input_cube = load_stac(
        url=url,
        spatial_extent=bounding_box,
        bands=["red", "nir"],
    ).isel({"x": slice(0, 20), "y": slice(0, 20)})

    # Test whether this works with different band names
    input_cube = input_cube.rename({"band": "b"})

    import dask.array as da

    numpy_data = _random_raster_data(input_cube.data.shape, dtype=np.float64)

    input_cube.data = da.from_array(numpy_data, chunks=("auto", "auto", "auto", -1))

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

    target_band = "yay"
    output_with_extra_dim = ndvi(input_cube, target_band=target_band)
    assert len(output_with_extra_dim.dims) == len(output.dims) + 1
    assert (
        len(output_with_extra_dim.coords[band_dim])
        == len(input_cube.coords[band_dim]) + 1
    )

    with pytest.raises(BandExists):
        output_with_extra_dim = ndvi(input_cube, target_band="time")
