def test_process_registry(process_registry):
    assert "max" in process_registry
    assert "_max" in process_registry

    assert not any(
        [
            process_id.startswith("_") or process_id.endswith("_")
            for process_id in process_registry.store.keys()
        ]
    )


def test_process_registry_aliases(process_registry):
    size_before = len(process_registry)

    assert "test_max" not in process_registry
    process_registry.add_alias("max", "test_max")
    assert "test_max" in process_registry
    assert process_registry["test_max"] == process_registry["max"]

    size_after = len(process_registry)
    assert size_after == size_before
