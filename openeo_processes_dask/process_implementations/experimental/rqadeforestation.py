from typing import Optional

import dask.array as da
from rqadeforestation import rqatrend

__all__ = ["rqadeforestation"]


def rqadeforestation(
    data,
    threshold: Optional[float] = 0.5,
    axis: Optional[int] = None,
):
    # directly call dask, no need to use UDF
    res = da.apply_along_axis(lambda x: rqatrend(x, threshold=threshold), axis, data)
    return res
