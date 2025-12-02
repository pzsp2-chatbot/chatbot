class CollectionAlreadyExistsError(Exception):
    """Exception raised when a collection already exists"""
    pass


class CollectionDoesNotExistError(Exception):
    """Exception raised when a collection does not exist"""
    pass


class DocumentDoesNotExistError(Exception):
    """Exception raised when a document with provided id does not exist"""
    pass


class InvalidDateFormatError(Exception):
    """Exception raised when a date format is invalid"""
    pass


class InputDataError(Exception):
    """Exception raised when input data is invalid"""
    pass
