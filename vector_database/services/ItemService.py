import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from vector_database.exceptions import CollectionDoesNotExistError, DocumentDoesNotExistError
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

    def delete_item(self, name: str, document_id: str):
        try:
            self.client.get_collection(collection_name=name)
        except Exception:
            raise CollectionDoesNotExistError(f"Collection '{name}' not found.")

        points, _ = self.client.scroll(collection_name=name, scroll_filter=Filter(must=[FieldCondition(
                        key="document_id", match=MatchValue(value=document_id))]), limit=1)
        if len(points) == 0:
            raise DocumentDoesNotExistError(f"Item with document id '{document_id}' not found.")

        self.client.delete(collection_name=name, points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])))

        return f"Item with document_id {document_id} deleted from collection {name}."
