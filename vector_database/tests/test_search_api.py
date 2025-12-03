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
                        "title": "Title1",
                        "created": 20250101,
                        "modified": 20250103,
                        "language": "en",
                        "doi": "example1",
                        "url": "https://url.org/example1",
                        "authors": ["Robert Smith", "Will Smith"],
                        "author_affiliations": ["WUT", "WU"],
                        "abstract": "First document",
                        "keywords": ["one", "two", "three"],
                        "document_id": "1"
                    }
                },
                {
                    "id": 2,
                    "vector": [0.5, 0.6, 0.7, 0.8],
                    "payload": {
                        "title": "Title2",
                        "created": 20250215,
                        "modified": 20250330,
                        "language": "en",
                        "doi": "example2",
                        "url": "https://url.org/example2",
                        "authors": ["John Doe"],
                        "author_affiliations": ["WUT", "WU"],
                        "abstract": "Second document",
                        "keywords": ["one", "four"],
                        "document_id": "2"
                    }
                },
                {
                    "id": 3,
                    "vector": [0.9, 0.6, 0.1, 0.8],
                    "payload": {
                        "title": "Title3",
                        "created": 20250115,
                        "modified": 20250330,
                        "language": "pl",
                        "doi": "example2",
                        "url": "https://url.org/example2",
                        "authors": ["Brad Pitt", "Will Smith"],
                        "author_affiliations": ["WUT", "AGH"],
                        "abstract": "Second document",
                        "keywords": ["six", "seven"],
                        "document_id": "3"
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
        "filter": {"title": "Title1", "starting_creation_date": "2025-01-01",
                   "ending_creation_date": "2025-01-05", "starting_modification_date": "2025-01-03",
                   "ending_modification_date": "2025-01-10", "language": "en", "doi": "example1",
                   "url": "https://url.org/example1", "authors": ["Robert Smith", "Will Smith"],
                   "author_affiliations": ["WUT", "WU"]},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["payload"]["authors"] == ["Robert Smith", "Will Smith"]


def test_search_success_no_filter(setup_collection):
    collection_name = setup_collection
    payload = {"vector": [0.1, 0.2, 0.3, 0.4], "top_k": 2, "filter": {}}
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert data["items"][0]["payload"]["authors"] == ["Robert Smith", "Will Smith"]


def test_search_success_filter_by_title(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.5, 0.6, 0.7, 0.8],
        "filter": {"title": "Title1"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["payload"]["title"] == "Title1"


def test_search_success_filter_by_modification_date(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.5, 0.6, 0.7, 0.8],
        "filter": {"starting_modification_date": "2025-01-30", "ending_modification_date": "2025-03-31"},
        "top_k": 3
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2


def test_search_success_filter_by_language(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.5, 0.6, 0.7, 0.8],
        "filter": {"language": "pl"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["payload"]["language"] == "pl"


def test_search_success_filter_by_doi(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.5, 0.6, 0.7, 0.8],
        "filter": {"doi": "example1"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["payload"]["doi"] == "example1"


def test_search_success_filter_by_url(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.5, 0.6, 0.7, 0.8],
        "filter": {"url": "https://url.org/example2"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["payload"]["url"] == "https://url.org/example2"


def test_search_success_filter_by_author_affiliations(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.5, 0.6, 0.7, 0.8],
        "filter": {"author_affiliations": ["WUT", "WU"]},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["payload"]["author_affiliations"] == ["WUT", "WU"]


def test_search_success_filter_by_authors(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.5, 0.6, 0.7, 0.8],
        "filter": {"authors": ["Will Smith"]},
        "top_k": 2
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert "Will Smith" in data["items"][0]["payload"]["authors"]


def test_search_success_filter_by_date_range(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.9, 0.6, 0.1, 0.8],
        "filter": {"starting_creation_date": "2025-01-10", "ending_creation_date": "2025-01-20"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["payload"]["authors"] == ["Brad Pitt", "Will Smith"]


def test_search_success_filter_author_and_date(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {"authors": ["Brad Pitt"], "starting_modification_date": "2025-03-01",
                    "ending_modification_date": "2025-04-01"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["payload"]["authors"] == ["Brad Pitt", "Will Smith"]


def test_search_success_no_results(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {"authors": ["Nonexistent Author", "Robert Smith"], "starting_date": "2025-01-01",
                   "ending_date": "2025-01-02"},
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
    assert data["detail"]["message"] == "Collection 'nonexistent_collection' not found."


def test_search_failed_invalid_vector_type(setup_collection):
    collection_name = setup_collection
    payload = {"vector": "test", "filter": {}, "top_k": 1}
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 422


def test_search_failed_filter_wrong_author_type(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {"authors": 123},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 422


def test_search_failed_filter_wrong_date_format(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {"authors": ["John Doe"], "starting_creation_date": "01-01-2025",
                   "ending_creation_date": "2025-01-31"},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 422


def test_search_failed_filter_wrong_range_type(setup_collection):
    collection_name = setup_collection
    payload = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "filter": {"starting_modification_date": 20250101, "ending_modification_date": 20250131},
        "top_k": 1
    }
    response = client.post(f"/collections/{collection_name}/search", json=payload)

    assert response.status_code == 422
