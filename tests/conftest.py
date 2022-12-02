import importlib
import inspect
import logging

import numpy as np
import pytest
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

from openeo_processes_dask.core import ProcessRegistry

logger = logging.getLogger(__name__)


@pytest.fixture
def random_data(size, dtype, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.integers(-100, 100, size=size)
    data = data.astype(dtype)
    return data


@pytest.fixture
def bounding_box(
    west=10.45, east=10.5, south=46.1, north=46.2, crs="EPSG:4326"
) -> BoundingBox:
    spatial_extent = {
        "west": west,
        "east": east,
        "south": south,
        "north": north,
        "crs": crs,
    }
    return BoundingBox.parse_obj(spatial_extent)


@pytest.fixture
def temporal_interval(interval=["2018-05-01", "2018-06-01"]) -> TemporalInterval:
    return TemporalInterval.parse_obj(interval)


@pytest.fixture
def process_registry() -> ProcessRegistry:
    standard_processes = [
        func
        for _, func in inspect.getmembers(
            importlib.import_module("openeo_processes_dask.process_implementations"),
            inspect.isfunction,
        )
    ]

    registry = ProcessRegistry()

    for process in standard_processes:
        registry[process.__name__] = process

    return registry
