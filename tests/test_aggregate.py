import numpy as np
import pytest
from openeo_pg_parser_networkx.pg_schema import TemporalInterval

from openeo_processes_dask.process_implementations.cubes.aggregate import (
    aggregate_temporal_period,
)
from openeo_processes_dask.process_implementations.math import mean
from tests.general_checks import general_output_checks
from tests.mockdata import create_fake_rastercube


@pytest.mark.parametrize("size", [(6, 5, 4, 4)])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize(
    "temporal_extent,period,expected",
    [
        (["2018-05-01", "2018-05-02"], "hour", 25),
        (["2018-05-01", "2018-06-01"], "day", 32),
        (["2018-05-01", "2018-06-01"], "week", 5),
        (["2018-05-01", "2018-06-01"], "month", 2),
        (["2018-01-01", "2018-12-31"], "season", 5),
        (["2018-01-01", "2018-12-31"], "year", 1),
    ],
)
def test_aggregate_temporal_period(
    temporal_extent, period, expected, bounding_box, random_data
):
    """"""
    bands = ["B02", "B03", "B04", "B08"]

    input_cube = create_fake_rastercube(
        data=random_data,
        spatial_extent=bounding_box,
        temporal_extent=TemporalInterval.parse_obj(temporal_extent),
        bands=bands,
    )
    output_cube = aggregate_temporal_period(
        data=input_cube, period=period, reducer=mean
    )

    general_output_checks(
        input_cube=input_cube,
        output_cube=output_cube,
        verify_attrs=True,
        verify_crs=True,
    )

    assert len(output_cube.t) == expected
    assert isinstance(output_cube.t.values[0], np.datetime64)
