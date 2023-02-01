import pytest
from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.core import process
from openeo_processes_dask.exceptions import ProcessParameterMissing


def test_process_decorator():
    def test_process(param3, param4, param5):
        return param3, param4, param5

    result = process(test_process)(
        1,
        5,
        param3=3,
        param4=ParameterReference(from_parameter="test_param_ref_4"),
        param5=ParameterReference(from_parameter="test_param_ref_5"),
        named_parameters={"test_param_ref_2": 2, "test_param_ref_4": 4},
        positional_parameters={"test_param_ref_5": 1},
    )
    assert result == (3, 4, 5)


def test_process_decorator_missing_parameter():
    def test_process(param1, param2=6):
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


def test_process_decorator_axis():
    def test_process(param1, param2=6, axis=-1):
        return param1, param2, axis

    result = process(test_process)(param1=1, param2=2)
    assert result == (1, 2, -1)

    def test_process_no_axis(param1, param2=6):
        return param1, param2

    result = process(test_process_no_axis)(param1=1, param2=2, axis=-1)
    assert result == (1, 2)
