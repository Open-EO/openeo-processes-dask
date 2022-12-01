import importlib
import inspect
import logging
from collections.abc import MutableMapping
from functools import wraps
from typing import Callable, Optional

from openeo_pg_parser_networkx.pg_schema import ParameterReference

from openeo_processes_dask.exceptions import ProcessParameterMissing
from openeo_processes_dask.process_implementations.cubes.utils import RENAME_DIMS

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

        # If necessary, rename dimension names here too!
        for k, v in resolved_kwargs.items():
            if k in ["dimension", "dim"] and v in RENAME_DIMS.keys():
                resolved_kwargs[k] = RENAME_DIMS[v]

        pretty_args = {k: type(v) for k, v in resolved_kwargs.items()}
        logger.warning(f"Running process {f.__name__}")
        logger.warning(f"kwargs: {pretty_args}")
        logger.warning("-" * 80)

        return f(*resolved_args, **resolved_kwargs)

    return wrapper


class ProcessRegistry(MutableMapping):
    """
    The process registry is basically a dictionary mapping from process_id to the callable implementation.
    It also allows registering aliases for process_ids.
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()  # type: dict[str, Callable]
        self.aliases = dict()  # type: dict[str, str]

        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key) -> Callable:
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        t_key = self._keytransform(key)
        decorated_value = process(value)
        self.store[t_key] = decorated_value

        # Update aliases if any exist
        if t_key in self.aliases:
            for alias in self.aliases[t_key]:
                self.store[alias] = decorated_value

    def __delitem__(self, key):
        t_key = self._keytransform(key)

        del self.store[t_key]

        # Update aliases if any exist
        if t_key in self.aliases:
            for alias in self.aliases[t_key]:
                del self.store[alias]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        """Some processes are prefixed with an underscore to prevent clashes with built-in names.
        These need to be stripped before being put into the registry."""
        return key.strip("_")

    def add_alias(self, process_id: str, alias: str):
        """
        Method to allow adding aliases to processes.
        This can be useful for not-yet standardised processes, where an OpenEO client might use a different process_id than the backend.
        """

        if process_id not in self.keys():
            raise ValueError(
                f"Could not add alias {alias} -> {process_id}, because process_id {process_id} was not found in the process registry."
            )

        self[self._keytransform(alias)] = self[self._keytransform(process_id)]
        self.aliases[self._keytransform(process_id)] = self._keytransform(alias)

        logger.debug(f"Added alias {alias} -> {process_id} to process registry.")


# This is not cool in most Python code, but I think it's fine here. It allows us to import and register new functions by just upgrading
# the process_implementation package, without adding it to this list here!
standard_processes = [
    func
    for _, func in inspect.getmembers(
        importlib.import_module("openeo_processes_dask.process_implementations"),
        inspect.isfunction,
    )
]

process_registry = ProcessRegistry()
for p in standard_processes:
    process_registry[p.__name__] = p
