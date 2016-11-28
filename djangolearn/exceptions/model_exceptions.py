
class NotMachineLearningModelException(Exception):
    """Raised when we're trying to store a non machine learning model."""
    pass


class FrameworkVersionCollisionException(Exception):
    """Raised when a stored model has different version from expected."""
    pass

class FeaturesVersionCollisionException(Exception):
    """Raised when a stored model has different version from expected."""
    pass


class ModelStorageException(Exception):
    """Raised when a model has a problem in storage."""
    pass


class ModelNotStoredException(ModelStorageException):
    """Raised when there is no stored model."""
    pass
