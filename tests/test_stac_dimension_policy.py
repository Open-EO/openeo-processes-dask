# tests/test_stac_dimension_policy.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pystac
import pytest

DEFAULT_OPENEO_DIMS: tuple[str, ...] = ("t", "bands", "y", "x")


def _read_stac(path: str) -> Union[pystac.Collection, pystac.Item]:
    obj = pystac.read_file(path)
    if not isinstance(obj, (pystac.Collection, pystac.Item)):
        raise TypeError(f"Expected STAC Collection or Item, got: {type(obj)}")
    return obj


def _get_cube_dimensions_mapping(
    stac_obj: Union[pystac.Collection, pystac.Item]
) -> dict[str, Any] | None:
    """
    Return dict stored under `cube:dimensions` (STAC Datacube extension) if present, else None.
    In pystac, extension fields are available under `.extra_fields`.
    """
    extra = getattr(stac_obj, "extra_fields", {}) or {}
    cube_dims = extra.get("cube:dimensions")
    return cube_dims if isinstance(cube_dims, dict) else None


def resolve_dimension_names(
    stac_obj: Union[pystac.Collection, pystac.Item]
) -> tuple[str, ...]:
    """
    Policy:
      - If `cube:dimensions` exists: enforce/use the dimension names as provided (keys)
      - Else: enforce default openEO dimension naming convention
    """
    cube_dims = _get_cube_dimensions_mapping(stac_obj)
    if cube_dims:
        return tuple(cube_dims.keys())
    return DEFAULT_OPENEO_DIMS


@pytest.mark.parametrize(
    "stac_path, expected_dims, must_have_cube_dims",
    [
        (
            "./tests/data/stac/s2_sample_dimension_policy_case_1.json",
            {"t", "bands", "y", "x"},
            False,
        ),
        (
            "./tests/data/stac/s2_sample_dimension_policy_case_2.json",
            {"t", "bands", "y", "x"},
            False,
        ),
        (
            "./tests/data/stac/s2_sample_dimension_policy_case_3.json",
            {"time", "band", "y", "x"},
            True,
        ),
    ],
)
def test_dimension_policy_from_collection_json(
    stac_path: str,
    expected_dims: set[str],
    must_have_cube_dims: bool,
) -> None:
    p = Path(stac_path)
    if not p.exists():
        pytest.fail(f"STAC JSON not found: {stac_path}")

    stac_obj = _read_stac(stac_path)
    cube_dims = _get_cube_dimensions_mapping(stac_obj)

    got = resolve_dimension_names(stac_obj)

    # Order-insensitive check
    assert set(got) == expected_dims, (
        f"Unexpected dimension names for {stac_path}\n"
        f"Expected set: {expected_dims}\n"
        f"Got:          {got} (set={set(got)})\n"
        f"cube:dimensions present: {cube_dims is not None}"
    )

    # Ensure the policy logic is exercised correctly
    if cube_dims is None:
        # No cube:dimensions -> must fall back to openEO defaults
        assert set(got) == set(DEFAULT_OPENEO_DIMS), (
            f"{stac_path} has no cube:dimensions but did not fall back to openEO defaults.\n"
            f"Got: {got}, expected defaults: {DEFAULT_OPENEO_DIMS}"
        )
    else:
        # cube:dimensions exists -> names must match cube:dimensions keys (order-insensitive)
        assert set(got) == set(cube_dims.keys()), (
            f"{stac_path} has cube:dimensions but resolved dims do not match it.\n"
            f"cube:dimensions keys: {tuple(cube_dims.keys())}\n"
            f"resolved dims:        {got}"
        )

    if must_have_cube_dims:
        assert cube_dims is not None, (
            f"{stac_path} is expected to define cube:dimensions (needed for {expected_dims}), "
            f"but cube:dimensions was missing."
        )


def test_default_dims_when_cube_dimensions_missing_synthetic_collection() -> None:
    """
    Control test: a minimal Collection without `cube:dimensions` must fall back to openEO defaults.
    """
    coll_dict = {
        "type": "Collection",
        "stac_version": "1.0.0",
        "id": "dummy",
        "description": "dummy",
        "license": "proprietary",
        "extent": {
            "spatial": {"bbox": [[0, 0, 1, 1]]},
            "temporal": {"interval": [["2020-01-01T00:00:00Z", None]]},
        },
        "links": [],
    }
    coll = pystac.Collection.from_dict(coll_dict)

    assert _get_cube_dimensions_mapping(coll) is None
    assert set(resolve_dimension_names(coll)) == set(DEFAULT_OPENEO_DIMS)
