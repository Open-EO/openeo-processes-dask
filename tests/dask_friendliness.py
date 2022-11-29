# from functools import wraps
from typing import Callable
# from distributed.diagnostics.memory_sampler import MemorySampler 


def check_dask_friendliness(test_f: Callable) -> bool:
    return NotImplemented()
    
    # TODO: Use distributed.MemorySampler to determine whether a process implementation has 
    # loaded any data into the dask cluster's memory
    
    # @wraps(test_f)
    # def wrapper(*args, **kwargs):
    #     MemorySampler()

    #     return test_f(*args, **kwargs)

    # return wrapper
