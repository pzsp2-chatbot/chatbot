import pytest
from fastapi.testclient import TestClient
from vector_database.main import app
from qdrant_client import QdrantClient
from vector_database.tests.conftest import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY
from qdrant_client.models import Distance, VectorParams

client = TestClient(app)
qdrant_client = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    api_key=QDRANT_API_KEY
)

@pytest.fixture
def create_and_clean_collection():
    collection_name = "test_collection"
    qdrant_client.create_collection(
        collection_name=collection_name, vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    yield collection_name
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        pass


def test_delete_collection_success(create_and_clean_collection):
    collection_name = create_and_clean_collection
    response = client.delete(f"/collections/{collection_name}")
    collections = qdrant_client.get_collections()

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["message"] == f"Deleted collection {collection_name}."
    assert len(collections.collections) == 0


def test_delete_collection_nonexistent(create_and_clean_collection):
    collection_name = "nonexistent_collection"
    response = client.delete(f"/collections/{collection_name}")
    collections = qdrant_client.get_collections()

    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "collection not found"
    assert data["message"] == f"Collection {collection_name} not found."
    assert len(collections.collections) == 1
