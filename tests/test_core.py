import importlib
import inspect

from openeo_processes_dask.core import ProcessRegistry


def test_process_registry():
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

    assert "max" in registry
    assert "_max" in registry

    assert not any(
        [
            process_id.startswith("_") or process_id.endswith("_")
            for process_id in registry.store.keys()
        ]
    )
