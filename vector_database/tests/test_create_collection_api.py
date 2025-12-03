import warnings

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

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
def cleanup_collection():
    collection_name = "test_collection"
    yield collection_name
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print("Failed to delete collection: " + str(e))


def test_create_collection_success(cleanup_collection):
    collection_name = cleanup_collection
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.post(
        "/collections", json={"name": collection_name, "vector_size": 1024}
    )
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["message"] == f"Collection '{collection_name}' created successfully."
    assert collections_before == collections_after - 1

    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    assert collection_info.config.params.vectors.size == 1024


def test_create_collection_failed_collection_exists(cleanup_collection):
    collection_name = cleanup_collection
    collections_before = len(qdrant_client.get_collections().collections)
    client.post("/collections", json={"name": collection_name, "vector_size": 1024})
    response = client.post(
        "/collections", json={"name": collection_name, "vector_size": 1024}
    )
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["status"] == "bad request"
    assert (
        data["detail"]["message"]
        == f"Collection with name '{collection_name}' already exists."
    )
    assert collections_before == collections_after - 1


def test_create_collection_failed_invalid_name_type_pydantic():
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.post("/collections", json={"name": 123, "vector_size": 1024})
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 422
    assert collections_before == collections_after


def test_create_collection_failed_too_short_name_pydantic():
    name = ""
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.post("/collections", json={"name": name, "vector_size": 1024})
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 422
    assert collections_before == collections_after


def test_create_collection_failed_too_long_name_pydantic():
    name = 65 * "a"
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.post("/collections", json={"name": name, "vector_size": 1024})
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 422
    assert collections_before == collections_after


def test_create_collection_failed_invalid_vector_size_type_pydantic(cleanup_collection):
    collection_name = cleanup_collection
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.post(
        "/collections", json={"name": collection_name, "vector_size": "abc"}
    )
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 422
    assert collections_before == collections_after


def test_create_collection_failed_too_short_vector_pydantic(cleanup_collection):
    collection_name = cleanup_collection
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.post(
        "/collections", json={"name": collection_name, "vector_size": 0}
    )
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 422
    assert collections_before == collections_after


def test_create_collection_failed_too_long_vector_pydantic(cleanup_collection):
    collection_name = cleanup_collection
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.post(
        "/collections", json={"name": collection_name, "vector_size": 1025}
    )
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 422
    assert collections_before == collections_after
