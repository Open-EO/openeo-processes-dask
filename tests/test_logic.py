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
