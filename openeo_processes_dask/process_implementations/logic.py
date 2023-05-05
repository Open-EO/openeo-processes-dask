from typing import Any

from numpy.typing import ArrayLike
from xarray.core.duck_array_ops import notnull


def is_valid(x: Any) -> ArrayLike:
    return notnull(x)
