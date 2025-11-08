import pytest
from fastapi.testclient import TestClient
from vector_database.main import app
from qdrant_client import QdrantClient

client = TestClient(app)
QDRANT_URL = "http://localhost:6333"
qdrant_client = QdrantClient(url=QDRANT_URL)

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
    response = client.post("/create_collection", json={"name": collection_name, "vector_size": 1024})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["message"] == f"Collection {collection_name} created successfully."

    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    assert collection_info.config.vectors["size"] == 1024

def test_create_collection_failed_collection_exists(cleanup_collection):
    collection_name = cleanup_collection
    client.post("/create_collection", json={"name": collection_name, "vector_size": 1024})
    response = client.post("/create_collection", json={"name": collection_name, "vector_size": 1024})

    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "bad request"
    assert data["message"] == f"Collection with name {collection_name} already exists."

def test_create_collection_failed_invalid_name_type_pydantic():
    response = client.post("/create_collection", json={"name": 123, "vector_size": 1024})
    assert response.status_code == 422

def test_create_collection_failed_too_short_name_pydantic():
    name = ""
    response = client.post("/create_collection", json={"name": name, "vector_size": 1024})
    assert response.status_code == 422

def test_create_collection_failed_too_long_name_pydantic():
    name = 65 * "a"
    response = client.post("/create_collection", json={"name": name, "vector_size": 1024})
    assert response.status_code == 422

def test_create_collection_failed_invalid_vector_size_type_pydantic(cleanup_collection):
    collection_name = cleanup_collection
    response = client.post("/create_collection", json={"name": collection_name, "vector_size": "abc"})
    assert response.status_code == 422

def test_create_collection_failed_too_short_vector_pydantic(cleanup_collection):
    collection_name = cleanup_collection
    response = client.post("/create_collection", json={"name": collection_name, "vector_size": 1023})
    assert response.status_code == 422

def test_create_collection_failed_too_long_vector_pydantic(cleanup_collection):
    collection_name = cleanup_collection
    response = client.post("/create_collection", json={"name": collection_name, "vector_size": 1025})
    assert response.status_code == 422
