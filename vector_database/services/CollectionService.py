from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from vector_database.exceptions import (
    CollectionAlreadyExistsError,
    CollectionDoesNotExistError,
)


class CollectionService:
    def __init__(self, client: QdrantClient):
        self.client = client

    def create_collection(self, name: str, vector_size: int):
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return f"Collection '{name}' created successfully."
        except Exception as e:
            if "exists" in str(e).lower():
                raise CollectionAlreadyExistsError(
                    f"Collection with name '{name}' already exists."
                )
            raise

    def delete_collection(self, name: str):
        try:
            self.client.get_collection(collection_name=name)
        except Exception:
            raise CollectionDoesNotExistError(f"Collection '{name}' not found.")
        self.client.delete_collection(collection_name=name)
        return f"Deleted collection '{name}'."
