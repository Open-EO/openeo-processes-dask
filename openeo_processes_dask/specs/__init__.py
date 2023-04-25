import builtins
import json
import keyword
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

json_path = Path(__file__).parent / "openeo-processes"
process_json_paths = [pg_path for pg_path in (json_path).glob("*.json")]

# Go through all the jsons in the top-level of the specs folder and add them to be importable from here
# E.g. from openeo_processes_dask.specs import *
# This is frowned upon in most python code, but I think here it's fine and allows a nice way of importing these

__all__ = []

for spec_path in process_json_paths:
    spec_json = json.load(open(spec_path))

    process_name = spec_json["id"]
    # Make sure we don't overwrite any builtins
    if spec_json["id"] in dir(builtins) or keyword.iskeyword(spec_json["id"]):
        process_name = "_" + spec_json["id"]

    locals()[process_name] = spec_json
    __all__.append(process_name)
