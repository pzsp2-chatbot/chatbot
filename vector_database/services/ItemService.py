import uuid
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

        qdrant_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())

        payload_with_id = dict(request.payload)
        payload_with_id["document_id"] = document_id
        self.client.upsert(collection_name=name, points=[{"id": qdrant_id, "vector": request.vector,
            "payload": payload_with_id}])

        return f"Item added to collection '{name}'."

    def delete_item(self, name: str):
        pass
