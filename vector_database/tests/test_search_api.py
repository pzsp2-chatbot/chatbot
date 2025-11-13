import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import VectorParams, Distance
from vector_database.main import app
from qdrant_client import QdrantClient
from vector_database.tests.conftest import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY
import warnings

warnings.filterwarnings("ignore", message="Api key is used with an insecure connection.")

client = TestClient(app)
qdrant_client = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    api_key=QDRANT_API_KEY
)


@pytest.fixture
def setup_collection():
    collection_name = "test_collection"
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        pass

    qdrant_client.create_collection(
        collection_name=collection_name, vectors_config=VectorParams(size=4, distance=Distance.COSINE)
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
                        "published_at": 20250101
                    }
                },
                {
                    "id": 2,
                    "vector": [0.5, 0.6, 0.7, 0.8],
                    "payload": {
                        "text": "Second document",
                        "author": "John Doe",
                        "published_at": 20250215
                    }
                },
                {
                    "id": 3,
                    "vector": [0.9, 0.6, 0.1, 0.8],
                    "payload": {
                        "text": "Third document",
                        "author": "Brad Pitt",
                        "published_at": 20250115
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
        "filter": {"author": "Robert Smith", "starting_date": "2025-01-01", "ending_date": "2025-01-31"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == 1


def test_search_success_no_filter(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "top_k": 2,
        "filter": {}
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert data["items"][0]["id"] == 1

def test_search_success_filter_by_author(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.5, 0.6, 0.7, 0.8],
        "filter": {"author": "John Doe"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == 2


def test_search_success_filter_by_date_range(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.9, 0.6, 0.1, 0.8],
        "filter": {"starting_date": "2025-01-10", "ending_date": "2025-01-20"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == 3


def test_search_success_filter_author_and_date(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {"author": "Brad Pitt", "starting_date": "2025-01-01", "ending_date": "2025-02-01"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == 3


def test_search_success_no_results(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {"author": "Nonexistent Author", "starting_date": "2025-01-01", "ending_date": "2025-01-02"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 0


def test_search_failed_nonexistent_collection():
    payload = {"vector": [0.1, 0.2, 0.3, 0.4], "filter": {}, "top_k": 1}
    response = client.post("/collections/nonexistent_collection/search", json=payload)

    assert response.status_code == 404
    data = response.json()
    assert data["detail"]["status"] == "not found"
    assert data["detail"]["message"] == f"Collection 'nonexistent_collection' not found."


def test_search_failed_invalid_vector_type(setup_collection):
    collection_name = setup_collection
    payload = {"vector": "test", "filter": {}, "top_k": 1}
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 422


def test_search_failed_filter_wrong_author_type(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {"author": 123},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 422


def test_search_failed_filter_wrong_date_format(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {"author": "John Doe", "starting_date": "01-01-2025", "ending_date": "2025-01-31"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 422


def test_search_failed_filter_wrong_range_type(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {"starting_date": 20250101, "ending_date": 20250131},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)
    assert response.status_code == 422
