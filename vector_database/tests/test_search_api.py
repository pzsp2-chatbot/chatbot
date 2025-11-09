import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import VectorParams, Distance

from vector_database.main import app
from qdrant_client import QdrantClient
from vector_database.tests.conftest import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY

client = TestClient(app)

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


def test_search_success(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {
            "must": [
                {"key": "author", "match": "Robert Smith"},
                {
                    "key": "published_at",
                    "range": {
                        "gte": "2025-01-01",
                        "lte": "2025-01-31"
                    }
                }
            ]
        },
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == 1


def test_search_success_no_filter(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "top_k": 2
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == 1

def test_search_success_filter_by_author(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.5, 0.6, 0.7, 0.8],
        "filter": {
            "must": [
                {"key": "author", "match": "John Doe"}
            ]
        },
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == 2


def test_search_success_filter_by_date_range(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.9, 0.6, 0.1, 0.8],
        "filter": {
            "must": [
                {
                    "key": "published_at",
                    "range": {
                        "gte": "2025-01-10",
                        "lte": "2025-01-20"
                    }
                }
            ]
        },
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == 3


def test_search_success_filter_author_and_date(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {
            "must": [
                {"key": "author", "match": "Brad Pitt"},
                {
                    "key": "published_at",
                    "range": {
                        "gte": "2025-01-01",
                        "lte": "2025-01-02"
                    }
                }
            ]
        },
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == 3


def test_search_success_no_results(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {
            "must": [
                {"key": "author", "match": "Nonexistent Author"}
            ]
        },
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 0


def test_search_failed_nonexistent_collection():
    payload = {"vector": [0.1, 0.2, 0.3, 0.4], "top_k": 1}
    response = client.post("/collections/nonexistent_collection/search", json=payload)

    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "collection not found"
    assert data["message"] == f"Collection with name nonexistent_collection not found."


def test_search_failed_invalid_vector_type(setup_collection):
    collection_name = setup_collection
    payload = {"vector": "test", "top_k": 1}
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 422


def test_search_failed_vector_wrong_size(setup_collection):
    collection_name = setup_collection
    payload = {"vector": [0.1], "top_k": 1}
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 422


def test_search_failed_filter_wrong_author_type(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {
            "must": [
                {"key": "author", "match": 123}
            ]
        },
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 422


def test_search_failed_filter_wrong_date_format(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {
            "must": [
                {
                    "key": "published_at",
                    "range": {
                        "gte": "01-01-2025",
                        "lte": "2025-01-31"
                    }
                }
            ]
        },
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 422


def test_search_failed_filter_wrong_range_type(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {
            "must": [
                {
                    "key": "published_at",
                    "range": {
                        "gte": 20250101,
                        "lte": 20250131
                    }
                }
            ]
        },
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 422
