from qdrant_client import QdrantClient
from vector_database.exceptions import CollectionDoesNotExistError
from vector_database.models import AddItemRequest


class ItemService:
    def __init__(self, client: QdrantClient):
        self.client = client

    def add_item(self, name: str, request: AddItemRequest):
        try:
            self.client.get_collection(collection_name=name)
        except Exception:
            raise CollectionDoesNotExistError(f"Collection '{name}' not found.")

        self.client.upload_collection(collection_name=name, vectors=[request.vector], payload=[request.metadata])

        return f"Item added to collection '{name}'."

    def delete_item(self, name: str):
        pass
