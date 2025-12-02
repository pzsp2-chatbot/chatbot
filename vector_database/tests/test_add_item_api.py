import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import Distance, VectorParams
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
def create_and_cleanup_collection():
    collection_name = "test_collection"
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception:
        pass

    qdrant_client.create_collection(
        collection_name=collection_name, vectors_config=VectorParams(size=4, distance=Distance.COSINE)
    )
    qdrant_client.upsert(collection_name=collection_name, points=[
        {
            "id": 1,
            "vector": [0.1, 0.2, 0.3, 0.4],
            "payload":
            {
                    "document_id": "13244433",
                    "title": "title1",
                    "created": "2024-11-01",
                    "modified": "2024-11-02",
                    "language": "en",
                    "doi": "doi1",
                    "url": "url1",
                    "authors": ["Robert Smith", "Will Smith"],
                    "author_affiliations": ["WUT", "WU"],
                    "abstract": "Sample test1",
                    "keywords": "IFS, Discrete Space"
            }
        }]
    )

    yield collection_name
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print("Failed to delete collection: " + str(e))


item = {
        "id": 1,
        "vector": [0.1, 0.1, 0.1, 0.1],
        "payload":
        {
                "title": "title2",
                "created": "2024-11-10",
                "modified": "2024-11-30",
                "language": "en",
                "doi": "doi2",
                "url": "url2",
                "authors": ["Robert Smith", "Will Smith"],
                "author_affiliations": ["WUT", "WU"],
                "abstract": "Sample test2",
                "keywords": "IFS, Discrete Space"
        }
}


def test_add_data_success(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["message"] == f"Item added to collection '{collection_name}'."
    assert point_count == 2


def test_add_data_failed_invalid_vector_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["vector"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_metadata_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"] = 1
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1

def test_add_data_failed_vector_too_short(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["vector"] = []
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1

def test_add_data_failed_vector_too_long(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["vector"] = [0.1] * 5
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1

def test_add_data_failed_empty_metadata(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"] = {}
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_date_format(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["modified"] = "2025-10-01"
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_no_document_id(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["document_id"] = ""
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_no_title(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["title"] = ""
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_no_created(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["created"] = ""
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_no_modified(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["modified"] = ""
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_no_language(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["language"] = ""
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_no_doi(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["doi"] = ""
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_no_url(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["url"] = ""
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_no_authors(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["authors"] = []
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_no_author_affiliations(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["author_affiliations"] = []
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_no_abstract(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["abstract"] = ""
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_wrong_title_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["title"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_wrong_created_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["created"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_wrong_modified_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["modified"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_wrong_language_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["language"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_wrong_doi_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["doi"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_wrong_url_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["url"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_wrong_authors_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["authors"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_wrong_affiliation_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["author_affiliations"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_wrong_abstract_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["abstract"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_wrong_keywords_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["keywords"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_invalid_document_id_type(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["document_id"] = 123
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1


def test_add_data_failed_creation_after_modification(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["created"] = "2024-02-20"
    item["payload"]["modified"] = "2024-02-19"
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 400
    assert point_count == 1


def test_add_data_failed_authors_and_affiliation_lengths_mismatch(create_and_cleanup_collection):
    collection_name = create_and_cleanup_collection
    item["payload"]["authors"] = ["author1", "author2"]
    item["payload"]["authors_affiliation"] = ["WUT"]
    response = client.post(f"/collections/{collection_name}/items", json=item)
    point_count = qdrant_client.count(collection_name=collection_name).count

    assert response.status_code == 422
    assert point_count == 1
