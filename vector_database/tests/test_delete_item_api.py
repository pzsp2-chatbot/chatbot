import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from vector_database.main import app
from qdrant_client import QdrantClient
from vector_database.tests.conftest import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY
import warnings

warnings.filterwarnings(
    "ignore", message="Api key is used with an insecure connection."
)

client = TestClient(app)
qdrant_client = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", api_key=QDRANT_API_KEY
)


@pytest.fixture
def setup_collection():
    collection_name = "test_collection"
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        pass

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": 1,
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "payload": {
                        "text": "First document",
                        "author": "Robert Smith",
                        "published_at": "2025-01-01",
                        "document_id": "10",
                    },
                },
                {
                    "id": 2,
                    "vector": [0.5, 0.6, 0.7, 0.8],
                    "payload": {
                        "text": "Second document",
                        "author": "John Doe",
                        "published_at": "2025-02-15",
                        "document_id": "11",
                    },
                },
                {
                    "id": 3,
                    "vector": [0.9, 0.6, 0.1, 0.8],
                    "payload": {
                        "text": "Third document",
                        "author": "Brad Pitt",
                        "published_at": "2025-01-15",
                        "document_id": "12",
                    },
                },
            ],
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
    document_id = str(11)

    response = client.delete(f"/collections/{collection_name}/items/{document_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert (
        data["message"]
        == f"Item with document_id {document_id} deleted from collection {collection_name}."
    )

    points, _ = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]
        ),
        limit=1,
    )

    assert len(points) == 0


def test_delete_item_nonexistent(setup_collection):
    collection_name = setup_collection
    document_id = str(999)

    response = client.delete(f"/collections/{collection_name}/items/{document_id}")

    assert response.status_code == 404
    data = response.json()
    assert data["detail"]["status"] == "not found"
    assert (
        data["detail"]["message"] == f"Item with document id '{document_id}' not found."
    )
