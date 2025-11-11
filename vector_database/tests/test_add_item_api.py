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
def create_and_cleanup_collection():
    collection_name = "test_collection"
    qdrant_client.create_collection(
        collection_name=collection_name, vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    yield collection_name
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print("Failed to delete collection: " + str(e))

def test_add_data_success(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": [0.1] * 1024, "metadata": {"text": "Hello world"}}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["message"] == f"Item added to collection '{collection_name}'."
    assert point_count == 1


def test_add_data_failed_invalid_vector_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": 123, "metadata": {"text": "Hello world"}}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 0

def test_add_data_failed_invalid_metadata_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": [0.1] * 1024, "metadata": 1}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 0

def test_add_data_failed_vector_too_short(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": [0.1] * 1023, "metadata": {"text": "Hello world"}}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 0

def test_add_data_failed_vector_too_long(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": [0.1] * 1025, "metadata": {"text": "Hello world"}}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 0

def test_add_data_failed_empty_metadata(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": [0.1] * 1025, "metadata": {}}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 0
