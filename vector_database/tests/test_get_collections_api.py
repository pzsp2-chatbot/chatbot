import pytest
from fastapi.testclient import TestClient
from vector_database.main import app
from qdrant_client import QdrantClient
from vector_database.tests.conftest import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY
from qdrant_client.models import Distance, VectorParams
import warnings

warnings.filterwarnings("ignore", message="Api key is used with an insecure connection.")

client = TestClient(app)
qdrant_client = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    api_key=QDRANT_API_KEY
)

@pytest.fixture
def create_and_clean_collection():
    collection_name = "test_collection"
    qdrant_client.create_collection(
        collection_name=collection_name, vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    yield collection_name
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        pass


def test_get_one_collection_success(create_and_clean_collection):
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.get("/collections")
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 200
    data = response.json()
    assert len(data["collections"]) == 1
    assert data["collections"][0] == "test_collection"
    assert collections_before == collections_after


def test_get_many_collections_success(create_and_clean_collection):
    qdrant_client.create_collection(
        collection_name="test_collection2", vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.get("/collections")
    collections_after = len(qdrant_client.get_collections().collections)
    qdrant_client.delete_collection(collection_name="test_collection2")

    assert response.status_code == 200
    data = response.json()
    assert len(data["collections"]) == 2
    assert collections_before == collections_after


def test_get_no_collections_success(create_and_clean_collection):
    qdrant_client.delete_collection(collection_name="test_collection")
    collections_before = len(qdrant_client.get_collections().collections)
    response = client.get("/collections")
    collections_after = len(qdrant_client.get_collections().collections)

    assert response.status_code == 200
    data = response.json()
    assert len(data["collections"]) == 0
    assert collections_before == collections_after
