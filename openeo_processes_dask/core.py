import importlib
import inspect
import logging
from functools import wraps
from typing import Optional

from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.exceptions import ProcessParameterMissing

logger = logging.getLogger(__name__)


def process(f):
    @wraps(f)
    def wrapper(*args, parameters: Optional[dict[str]] = None, **kwargs):
        if parameters is None:
            parameters = {}

        resolved_args = []
        for arg in args:
            if isinstance(arg, ParameterReference):
                if arg.from_parameter in parameters:
                    resolved_args.append(parameters[arg.from_parameter])
                else:
                    raise ProcessParameterMissing(
                        f"Error: Process Parameter {arg.from_parameter} was missing for process {f.__name__}"
                    )
            else:
                resolved_args.append(arg)

        resolved_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, ParameterReference):
                if v.from_parameter in parameters:
                    resolved_kwargs[k] = parameters[v.from_parameter]
                else:
                    raise ProcessParameterMissing(
                        f"Error: Process Parameter {v.from_parameter} was missing for process {f.__name__}"
                    )
            else:
                resolved_kwargs[k] = v

        pretty_args = {k: type(v) for k, v in resolved_kwargs.items()}
        logger.warning(f"Running process {f.__name__}")
        logger.warning(f"kwargs: {pretty_args}")
        logger.warning("-" * 80)

        return f(*resolved_args, **resolved_kwargs)

    return wrapper


# This is not cool in most Python code, but I think it's fine here. It allows us to import and register new functions by just upgrading
# the process_implementation package, without adding it to this list here!
standard_processes = [
    func
    for _, func in inspect.getmembers(
        importlib.import_module("openeo_processes_dask.process_implementations"),
        inspect.isfunction,
    )
]
