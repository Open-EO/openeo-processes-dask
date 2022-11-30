from functools import partial

import numpy as np
import pytest
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.core import process_registry
from openeo_processes_dask.process_implementations.cubes.apply import apply
from openeo_processes_dask.process_implementations.math import add
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_apply(temporal_interval, bounding_box, random_data):
    input_cube = create_fake_rastercube(
        data=random_data,
        spatial_extent=bounding_box,
        temporal_extent=temporal_interval,
        bands=["B02", "B03", "B04", "B08"],
    )

    _process = partial(
        process_registry["add"], y=1, x=ParameterReference(from_parameter="x")
    )

    output_cube = apply(data=input_cube, process=_process)

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )
