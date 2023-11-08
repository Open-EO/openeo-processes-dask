import importlib
import inspect
import logging
from functools import wraps
from typing import Optional

from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.process_implementations.exceptions import (
    ProcessParameterMissing,
)

logger = logging.getLogger(__name__)


def process(f):
    """
    The `@process` decorator resolves ParameterReferences and is expected to be wrapped around all processes.
    This is necessary because openeo_pg_parser_networkx parses and injects raw ParameterReference objects as input to each process node.
    However the process implementations in openeo-processes-dask cannot handle these and require the actual objects that the ParameterReferences refer to.
    This decorator ensures that incoming ParameterReferences are resolved to the actual inputs before being passed into the process implementations.
    """

    @wraps(f)
    def wrapper(
        *args,
        positional_parameters: Optional[dict[int]] = None,
        named_parameters: Optional[dict[str]] = None,
        **kwargs,
    ):
        # Need to transform this from a tuple to a list to be able to delete from it.
        args = list(args)

        # Some processes like `apply` cannot pass a parameter for a child-process using kwargs, but only by position.
        # E.g. `apply` passes the data to apply over as a parameter `x`, but the implementation with `apply_ufunc`
        # does not allow naming this parameter `x`.
        # The `positional_parameters` dictionary allows parent ("callback") processes to assign names to positional arguments it passes on.
        if positional_parameters is None:
            positional_parameters = {}

        if named_parameters is None:
            named_parameters = {}

        resolved_args = []
        resolved_kwargs = {}

        # If an arg is specified in positional_parameters, put the correct key-value pair into named_parameters
        for arg_name, i in positional_parameters.items():
            named_parameters[arg_name] = args[i]

        for arg in args:
            if isinstance(arg, ParameterReference):
                if arg.from_parameter in named_parameters:
                    resolved_args.append(named_parameters[arg.from_parameter])
                else:
                    raise ProcessParameterMissing(
                        f"Error: Process Parameter {arg.from_parameter} was missing for process {f.__name__}"
                    )

        for k, arg in kwargs.items():
            if isinstance(arg, ParameterReference):
                if arg.from_parameter in named_parameters:
                    resolved_kwargs[k] = named_parameters[arg.from_parameter]
                else:
                    raise ProcessParameterMissing(
                        f"Error: Process Parameter {arg.from_parameter} was missing for process {f.__name__}"
                    )
            else:
                resolved_kwargs[k] = arg

        special_args = ["axis", "keepdims", "source_transposed_axis"]
        # Remove 'axis' and keepdims parameter if not expected in function signature.
        for arg in special_args:
            if arg not in inspect.signature(f).parameters:
                resolved_kwargs.pop(arg, None)

        pretty_args = {k: repr(v)[:80] for k, v in resolved_kwargs.items()}
        if hasattr(f, "__name__"):
            logger.info(f"Running process {f.__name__}")
            logger.debug(
                f"Running process {f.__name__} with resolved parameters: {pretty_args}"
            )

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
