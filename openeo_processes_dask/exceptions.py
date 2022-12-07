class DimensionNotAvailable(Exception):
    def __init__(self, msg):
        self.message = "A dimension with the specified name does not exist."

    def __str__(self):
        return self.message


class DimensionLabelCountMismatch(Exception):
    def __init__(self, msg):
        self.message = (
            "The number of dimension labels exceeds one, which requires a reducer."
        )

    def __str__(self):
        return self.message


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
        self.message = (
            "The number of dimensions must be reduced to three for `aggregate_spatial`."
        )

    def __str__(self):
        return self.message


class ProcessParameterMissing(Exception):
    pass


class ModelNotFoundException(Exception):
    pass


class DimensionNotAvailable(Exception):
    pass
