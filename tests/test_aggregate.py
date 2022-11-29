import numpy as np
import pytest
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

from openeo_processes_dask.process_implementations.cubes.aggregate import (
    aggregate_temporal_period,
)
from openeo_processes_dask.process_implementations.math import mean
from tests.mockdata import generate_fake_rastercube


@pytest.mark.parametrize(
    "incoming_temporal_extent,period,expected",
    [
        (["2018-05-01", "2018-05-02"], "hour", 25),
        (["2018-05-01", "2018-06-01"], "day", 32),
        (["2018-05-01", "2018-06-01"], "week", 5),
        (["2018-05-01", "2018-06-01"], "month", 2),
        (["2018-01-01", "2018-12-31"], "season", 5),
        (["2018-01-01", "2018-12-31"], "year", 1),
    ],
)
def test_aggregate_temporal_period(incoming_temporal_extent, period, expected):
    """"""
    spatial_extent = {
        "west": 10.45,
        "east": 10.5,
        "south": 46.1,
        "north": 46.2,
        "crs": "EPSG:4326",
    }
    bands = ["B02", "B03", "B04", "B08"]
    temporal_extent = incoming_temporal_extent

    test_data = generate_fake_rastercube(
        "test",
        spatial_extent=BoundingBox.parse_obj(spatial_extent),
        temporal_extent=TemporalInterval.parse_obj(temporal_extent),
        bands=bands,
    )
    aggregated = aggregate_temporal_period(data=test_data, period=period, reducer=mean)

    assert len(aggregated.t) == expected
    assert type(aggregated.t.values[0]) == np.datetime64
