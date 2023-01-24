from functools import partial

import numpy as np
import pytest
import xarray as xr
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.logic import *
from tests.general_checks import assert_numpy_equals_dask_numpy, general_output_checks
from tests.mockdata import create_fake_rastercube


def test_and_():
    assert and_(True, True)
    assert not and_(True, False)
    assert and_(True, None) is None
    assert (
        and_(x=[True, True, False, False], y=[True, False, False, np.nan])
        == [True, False, False, False]
    ).all()


def test_or_():
    assert or_(True, True)
    assert or_(True, False)
    assert or_(True, None)
    assert or_(False, None) is None
    assert (
        or_(x=[True, True, False, False], y=[True, False, False, np.nan])
        == [True, True, False, False]
    ).all()


def test_xor():
    assert not xor(True, True)
    assert xor(True, False)
    assert not xor(False, None)
    assert (
        xor(x=[True, True, False, False], y=[True, False, False, np.nan])
        == [False, True, False, False]
    ).all()


print(test_xor())
print(xor(True, None))
