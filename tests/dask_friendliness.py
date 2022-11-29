from functools import wraps
from distributed.diagnostics.memory_sampler import MemorySampler 



def check_dask_friendliness(test_f) -> bool:
    @wraps(test_f)
    def wrapper(*args, **kwargs):
        MemorySampler()

        return test_f(*args, **kwargs)

    return wrapper


