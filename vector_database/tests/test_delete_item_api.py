import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import Distance, VectorParams

from vector_database.main import app
from qdrant_client import QdrantClient

from vector_database.tests.conftest import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY

client = TestClient(app)
QDRANT_URL = "http://localhost:6333"
qdrant_client = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    api_key=QDRANT_API_KEY
)

@pytest.fixture
def setup_collection():
    collection_name = "test_collection"
    qdrant_client.create_collection(
        collection_name=collection_name, vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": 1,
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "metadata": {
                        "text": "First document",
                        "author": "Robert Smith",
                        "published_at": "2025-01-01"
                    }
                },
                {
                    "id": 2,
                    "vector": [0.5, 0.6, 0.7, 0.8],
                    "metadata": {
                        "text": "Second document",
                        "author": "John Doe",
                        "published_at": "2025-02-15"
                    }
                },
                {
                    "id": 3,
                    "vector": [0.9, 0.6, 0.1, 0.8],
                    "metadata": {
                        "text": "Third document",
                        "author": "Brad Pitt",
                        "published_at": "2025-01-15"
                    }
                }
            ]
        )
    except Exception as e:
        print("Failed to setup collection: " + str(e))
    yield collection_name
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print("Failed to delete collection: " + str(e))

def test_delete_item_success(setup_collection):
    collection_name = setup_collection
    item_id = 1

    response = client.delete(f"/collections/{collection_name}/items/{item_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["message"] == f"Item with id {item_id} deleted from collection {collection_name}."

    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    assert collection_info.result.points_count == 2

def test_delete_item_nonexistent(setup_collection):
    collection_name  = setup_collection
    item_id = 999

    response = client.delete(f"/collections/{collection_name}/items/{item_id}")

    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "item not found"
    assert data["message"] == f"Item with id {item_id} not found."

def test_delete_item_invalid_id_type(setup_collection):
    collection_name = setup_collection
    item_id =  "abc"

    response = client.delete(f"/collections/{collection_name}/items/{item_id}")

    assert response.status_code == 422