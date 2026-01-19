import os
import shutil
import json
import pytest
from embeddings.infrastructure.json_loader import JSONArticleLoader

TEST_FOLDER = "tests/tmp_json_loader"


@pytest.fixture(scope="module")
def setup_json():
    os.makedirs(TEST_FOLDER, exist_ok=True)

    dummy = {
        "id": "TEST123",
        "title": "Test Article",
        "created": "2025-01-01T10:00:00",
        "modified": "2025-01-02T12:00:00",
        "language": "en",
        "authors": [{"full_name": "Alice", "affiliation": "Uni"}],
        "abstract": "Dummy abstract",
        "keywords": "test; pytest; loader",
        "doi": "10.1234/testdoi",
        "url": "https://example.com",
    }

    with open(os.path.join(TEST_FOLDER, "dummy.json"), "w", encoding="utf-8") as f:
        json.dump(dummy, f)

    yield

    shutil.rmtree(TEST_FOLDER)


def test_loader_reads_json(setup_json):
    loader = JSONArticleLoader(TEST_FOLDER)
    articles = loader.load_all()

    assert len(articles) == 1

    article = articles[0]

    assert article.id == "TEST123"
    assert article.title == "Test Article"
    assert article.language == "en"
    assert article.abstract == "Dummy abstract"
    assert article.keywords == ["test", "pytest", "loader"]

    assert len(article.authors) == 1
    assert article.authors[0].full_name == "Alice"
    assert article.authors[0].affiliation == "Uni"

    text = article.to_text()
    assert "Title: Test Article" in text
    assert "Authors: Alice" in text
