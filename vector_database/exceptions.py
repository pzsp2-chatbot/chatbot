class CollectionAlreadyExistsError(Exception):
    """Exception raised when a collection already exists"""
    pass


class CollectionDoesNotExistError(Exception):
    """Exception raised when a collection does not exist"""
    pass
