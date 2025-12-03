from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from vector_database.exceptions import (
    CollectionAlreadyExistsError,
    CollectionDoesNotExistError,
)
from vector_database.models import CreateCollectionRequest


class CollectionService:
    def __init__(self, client: QdrantClient):
        self.client = client

    def get_collections(self) -> list[str]:
        response = self.client.get_collections()
        formatted_collections = [collection.name for collection in response.collections]

        return formatted_collections

    def create_collection(self, request: CreateCollectionRequest) -> str:
        try:
            self.client.create_collection(
                collection_name=request.name,
                vectors_config=VectorParams(
                    size=request.vector_size, distance=Distance.COSINE
                ),
            )
            return f"Collection '{request.name}' created successfully."

        except Exception as e:
            if "exists" in str(e).lower():
                raise CollectionAlreadyExistsError(
                    f"Collection with name '{request.name}' already exists."
                )
            raise

    def delete_collection(self, name: str) -> str:
        try:
            self.client.get_collection(collection_name=name)
        except Exception:
            raise CollectionDoesNotExistError(f"Collection '{name}' not found.")

        self.client.delete_collection(collection_name=name)
        return f"Deleted collection '{name}'."
