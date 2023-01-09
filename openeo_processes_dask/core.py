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
    def wrapper(
        *args,
        positional_parameters: Optional[dict[int]] = None,
        named_parameters: Optional[dict[str]] = None,
        **kwargs,
    ):
        if positional_parameters is None:
            positional_parameters = {}

        if named_parameters is None:
            named_parameters = {}

        resolved_args = []
        for arg in args:
            if isinstance(arg, ParameterReference):
                if arg.from_parameter in positional_parameters:
                    i = positional_parameters[arg.from_parameter]
                    # Take the parameter from the provided index of *args and resolve it
                    resolved_args.append(args[i])
                elif arg.from_parameter in named_parameters:
                    resolved_args.append(named_parameters[arg.from_parameter])
                else:
                    raise ProcessParameterMissing(
                        f"Error: Process Parameter {arg.from_parameter} was missing for process {f.__name__}"
                    )
            else:
                resolved_args.append(arg)

        resolved_kwargs = {}
        for k, arg in kwargs.items():
            if isinstance(arg, ParameterReference):
                if arg.from_parameter in named_parameters:
                    resolved_kwargs[k] = named_parameters[arg.from_parameter]
                elif arg.from_parameter in positional_parameters:
                    # This will have already been passed through from the first loop
                    pass
                else:
                    raise ProcessParameterMissing(
                        f"Error: Process Parameter {arg.from_parameter} was missing for process {f.__name__}"
                    )
            else:
                resolved_kwargs[k] = arg

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
