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
