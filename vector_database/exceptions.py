class CollectionAlreadyExistsError(Exception):
    """Exception raised when a collection already exists"""
    pass


class CollectionDoesNotExistError(Exception):
    """Exception raised when a collection does not exist"""
    pass


class DocumentDoesNotExistError(Exception):
    """Exception raised when a document with provided id does not exist"""
    pass
