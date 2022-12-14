class OpenEOException(Exception):
    pass


class DimensionLabelCountMismatch(OpenEOException):
    pass


class ArrayElementParameterConflict(OpenEOException):
    pass


class ArrayElementParameterMissing(OpenEOException):
    pass


class ArrayNotLabeled(OpenEOException):
    pass


class ArrayElementNotAvailable(OpenEOException):
    pass


class TooManyDimensions(OpenEOException):
    pass


class ProcessParameterMissing(OpenEOException):
    pass


class ModelNotFoundException(OpenEOException):
    pass


class DimensionNotAvailable(OpenEOException):
    pass


class OverlapResolverMissing(Exception):
    def __init__(self, msg):
        self.message = (
            "Overlapping data cubes, but no overlap resolver has been specified."
        )

    def __str__(self):
        return self.message
