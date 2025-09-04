from typing import Optional
import ctypes as ct
import numpy as np
import dask.array as da

from numpy.typing import ArrayLike
import rqadeforestation
from rqadeforestation import rqatrend

import os

__all__ = ["rqa"]


class MallocVector(ct.Structure):
    _fields_ = [("pointer", ct.c_void_p),
                ("length", ct.c_int64),
                ("s1", ct.c_int64)]

def mvptr(A):
    ptr = A.ctypes.data_as(ct.c_void_p)
    a = MallocVector(ptr, ct.c_int64(A.size), ct.c_int64(A.shape[0]))
    return ct.byref(a)


# download so file at https://github.com/EarthyScience/RQADeforestation.py/archive/refs/heads/main.zip
lib = ct.CDLL("./.venv/lib/python3.11/site-packages/rqadeforestation/lib/rqatrend.so")
lib.rqatrend.argtypes = (ct.POINTER(MallocVector), ct.c_double, ct.c_int64, ct.c_int64)
lib.rqatrend.restype = ct.c_double


def f(array: np.ndarray):
    y_ptr = mvptr(array)
    res = lib.rqatrend(y_ptr, 0.5, 10, 1)
    return res


def rqa(data, axis: Optional[int] = None,):
    res = da.apply_along_axis(f, axis=axis, arr=data, dtype=np.float64)
    return da.array(np.array(res)) # rqatrend(data, 0.5, 10, 1)