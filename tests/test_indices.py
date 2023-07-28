import numpy as np

from openeo_processes_dask.process_implementations.cubes.indices import ndvi
from openeo_processes_dask.process_implementations.cubes.load import load_stac
from tests.conftest import _random_raster_data


def test_ndvi(bounding_box):
    url = "./tests/data/stac/s2_l2a_test_item.json"
    input_cube = load_stac(
        url=url,
        spatial_extent=bounding_box,
        bands=["red", "nir"],
    )

    import dask.array as da

    numpy_data = _random_raster_data(input_cube.data.shape, dtype=np.float64)

    input_cube.data = da.from_array(numpy_data, chunks=("auto", "auto", "auto", -1))

    output = ndvi(input_cube)
