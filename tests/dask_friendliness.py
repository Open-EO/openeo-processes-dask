from functools import wraps
from time import sleep
from typing import Callable

import dask
import dask.distributed
import numpy as np
from dask.distributed import Client
from distributed.diagnostics.memory_sampler import MemorySampler

# This is a pytest plugin that adds a pytest_runtest_call hook to run the memory sampler around the test


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "check_dask_friendliness: mark test to check dask friendliness"
    )


def pytest_runtest_call(item) -> None:
    dask_friendliness_enabled = False

    for marker in item.own_markers:
        if marker.name == "check_dask_friendliness":
            dask_friendliness_enabled = True
            break

    if dask_friendliness_enabled:
        with Client():
            ms = MemorySampler()
            with ms.sample("mem_usage", interval=0.01):
                # Need to wait a bit for the sampler to catch the activity
                sleep(0.5)
                item.runtest()
                sleep(0.5)

            ms_pandas = ms.to_pandas()
            mem_usage_range = np.ptp(ms_pandas["mem_usage"])
            assert mem_usage_range == 0
    else:
        with Client():
            item.runtest()
        print("yo")


def check_dask_friendliness(test_f: Callable):

    # TODO: Use distributed.MemorySampler to determine whether a process implementation has
    # loaded any data into the dask cluster's memory

    @wraps(test_f)
    def wrapper(*args, **kwargs):
        if not dask.distributed.default_client():
            return test_f(*args, **kwargs)
        client = dask.distributed.default_client()
        ms = MemorySampler()
        with ms.sample("mem_usage"):
            result = test_f(*args, **kwargs)
        ms_pandas = ms.to_pandas()
        mem_usage_range = np.ptp(ms_pandas["mem_usage"])
        assert mem_usage_range == 0

        return result

    return wrapper
