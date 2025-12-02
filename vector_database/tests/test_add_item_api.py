import warnings

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from vector_database.main import app
from vector_database.tests.conftest import QDRANT_API_KEY, QDRANT_HOST, QDRANT_PORT

warnings.filterwarnings(
    "ignore", message="Api key is used with an insecure connection."
)

client = TestClient(app)
qdrant_client = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", api_key=QDRANT_API_KEY
)


@pytest.fixture
def create_and_cleanup_collection():
    collection_name = "test_collection"
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        pass

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
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
                },
            }
        ],
    )

    yield collection_name
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print("Failed to delete collection: " + str(e))


def test_add_data_success(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": [0.1] * 4, "payload": {"text": "Hello world"}}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["message"] == f"Item added to collection '{collection_name}'."
    assert point_count == 2


def test_add_data_failed_invalid_vector_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": 123, "payload": {"text": "Hello world"}}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_metadata_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": [0.1] * 4, "payload": 1}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_vector_too_short(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": [], "payload": {"text": "Hello world"}}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_vector_too_long(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": [0.1] * 1025, "payload": {"text": "Hello world"}}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_empty_metadata(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {"vector": [0.1] * 4, "payload": {}}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_date_format(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item = {
        "vector": 123,
        "payload": {
            "text": "text",
            "author": "Robert Smith",
            "published_at": "2025-10-01",
        },
    }
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1
