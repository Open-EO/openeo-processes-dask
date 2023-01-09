from functools import partial

import numpy as np
import pytest

from openeo_processes_dask.process_implementations.cubes.reduce import reduce_dimension
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


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

    _process = partial(process_registry["mean"], ignore_nodata=True)

    output_cube = reduce_dimension(data=input_cube, reducer=_process, dimension="t")
    # output_cube.compute()

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=False,
        verify_crs=True,
    )
