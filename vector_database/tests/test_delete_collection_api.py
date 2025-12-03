import pytest
from fastapi.testclient import TestClient
from vector_database.main import app
from qdrant_client import QdrantClient
from vector_database.tests.conftest import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY
from qdrant_client.models import Distance, VectorParams
import warnings

warnings.filterwarnings(
    "ignore", message="Api key is used with an insecure connection."
)

client = TestClient(app)
qdrant_client = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", api_key=QDRANT_API_KEY
)


@pytest.fixture
def create_and_clean_collection():
    collection_name = "test_collection"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    yield collection_name
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        pass


def test_delete_collection_success(create_and_clean_collection):
    collection_name = create_and_clean_collection
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.delete(f"/collections/{collection_name}")
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["message"] == f"Deleted collection '{collection_name}'."
    assert collections_before == collections_after + 1


def test_delete_collection_nonexistent(create_and_clean_collection):
    collection_name = "nonexistent_collection"
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.delete(f"/collections/{collection_name}")
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 404
    data = response.json()
    assert data["detail"]["status"] == "not found"
    assert data["detail"]["message"] == f"Collection '{collection_name}' not found."
    assert collections_before == collections_after
