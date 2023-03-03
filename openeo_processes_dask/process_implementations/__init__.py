import logging

logger = logging.getLogger(__name__)


from .arrays import *
from .comparison import *
from .cubes import *
from .logic import *
from .math import *

try:
    from .ml import *
except ImportError as e:
    logger.warning("Could not load ML processes due to missing dependencies: ", e)

import rioxarray as rio  # Required for the .rio accessor on xarrays.

import openeo_processes_dask.process_implementations.cubes._xr_interop
