import ctypes as ct
import os
import sys
from typing import Optional

import dask.array as da
import numpy as np
import rqadeforestation
from numpy.typing import ArrayLike
from rqadeforestation import rqatrend

__all__ = ["rqa"]


class MallocVector(ct.Structure):
    _fields_ = [("pointer", ct.c_void_p), ("length", ct.c_int64), ("s1", ct.c_int64)]


def mvptr(A):
    ptr = A.ctypes.data_as(ct.c_void_p)
    a = MallocVector(ptr, ct.c_int64(A.size), ct.c_int64(A.shape[0]))
    return ct.byref(a)


# download so file at https://github.com/EarthyScience/RQADeforestation.py/archive/refs/heads/main.zip
version = sys.version[:4]
lib = ct.CDLL(
    f"./.venv/lib/python{version}/site-packages/rqadeforestation/lib/rqatrend.so"
)
lib.rqatrend.argtypes = (ct.POINTER(MallocVector), ct.c_double, ct.c_int64, ct.c_int64)
lib.rqatrend.restype = ct.c_double


def rqa_vector(array: np.ndarray, threshold: float = 0.5) -> float:
    y_ptr = mvptr(array)
    res = lib.rqatrend(y_ptr, threshold, 10, 1)
    return res


def rqadeforestation(
    data,
    threshold: float,
    axis: Optional[int] = None,
):
    # allow reducer without UDF
    res = da.apply_along_axis(
        rqa_vector, axis=axis, arr=data, dtype=np.float64, threshold=threshold
    )
    return da.array(np.array(res))
