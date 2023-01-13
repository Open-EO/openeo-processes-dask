import pytest
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.core import process
from openeo_processes_dask.exceptions import ProcessParameterMissing


def test_process_decorator():
    def test_process(param1, param2, param3, param4, param5):
        return param1, param2, param3, param4, param5

    result = process(test_process)(
        1,
        ParameterReference(from_parameter="test_param_ref_2"),
        5,
        param3=3,
        param4=ParameterReference(from_parameter="test_param_ref_4"),
        named_parameters={"test_param_ref_2": 2, "test_param_ref_4": 4},
        positional_parameters={"param5": 2},
    )
    assert result == (1, 2, 3, 4, 5)


def test_process_decorator_missing_parameter():
    def test_process(param1, param2=6, **kwarg):
        return param1 * param2

    with pytest.raises(ProcessParameterMissing):
        process(test_process)(
            param1=ParameterReference(from_parameter="test_param_ref"),
            parameters={"wrong_param": 2},
        )

    with pytest.raises(ProcessParameterMissing):
        process(test_process)(
            ParameterReference(from_parameter="test_param_ref"),
            parameters={"wrong_param": 2},
        )
