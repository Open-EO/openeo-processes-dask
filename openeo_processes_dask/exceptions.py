class DimensionNotAvailable(Exception):
    pass


class ArrayElementParameterConflict(Exception):
    pass


class ArrayElementParameterMissing(Exception):
    pass


class ArrayNotLabeled(Exception):
    pass


class ArrayElementNotAvailable(Exception):
    pass


class TooManyDimensions(Exception):
    def __init__(self, msg):
        self.message = "The number of dimensions must be reduced to three for `aggregate_spatial`."

    def __str__(self):
        return self.message


class ProcessParameterMissing(Exception):
    pass


class ModelNotFoundException(Exception):
    pass


class DimensionNotAvailable(Exception):
    pass
