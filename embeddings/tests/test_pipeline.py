import pytest
from unittest.mock import MagicMock, patch
from embeddings.infrastructure.json_loader import JSONArticleLoader
from embeddings.infrastructure.openai_embedder import OpenAIEmbedder
from embeddings.infrastructure.hf_embedder import HFEmbedder
from embeddings.infrastructure.dummy_embedder import DummyEmbedder
from embeddings.pipeline import EmbeddingPipeline
from embeddings.models.author import Author
from embeddings.models.article import Article
import tempfile
import json
from pathlib import Path
import numpy as np

@pytest.fixture
def setup_pipeline_json():
    def _create(num_articles):
        folder = tempfile.TemporaryDirectory()
        for i in range(num_articles):
            article = {
                "id": f"TEST{i}",
                "title": f"Test Article {i}",
                "created": "2025-01-01T10:00:00",
                "modified": "2025-01-02T12:00:00",
                "language": "en",
                "authors": [{"full_name": f"Author {i}", "affiliation": "Uni"}],
                "abstract_en": f"Dummy abstract {i}",
                "abstract_pl": None,
                "doi": f"10.1234/testdoi{i}",
                "url": f"https://example.com/article_{i}"
            }
            file_path = Path(folder.name) / f"article_{i}.json"
            file_path.write_text(json.dumps(article), encoding="utf-8")
        return folder
    return _create

@pytest.mark.parametrize("num_articles, vector_size", [(1, 3), (3, 5), (5, 10)])
def test_pipeline_dummy_various(setup_pipeline_json, num_articles, vector_size):
    folder = setup_pipeline_json(num_articles)
    loader = JSONArticleLoader(folder.name)
    
    class DummyEmbedder:
        def __init__(self, vector_size):
            self.vector_size = vector_size
        def embed(self, texts):
            return [list(np.ones(self.vector_size)) for _ in texts]
    
    embedder = DummyEmbedder(vector_size=vector_size)
    pipeline = EmbeddingPipeline(loader, embedder)
    
    ids, embeddings, payloads = pipeline.run()
    
    assert len(ids) == num_articles
    assert len(embeddings) == num_articles
    assert all(len(vec) == vector_size for vec in embeddings)
    assert len(payloads) == num_articles

@pytest.mark.parametrize("num_articles", [1, 3])
def test_pipeline_openai_mock_various(setup_pipeline_json, num_articles):
    folder = setup_pipeline_json(num_articles)
    loader = JSONArticleLoader(folder.name)
    
    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.4, 0.5, 0.6])] * num_articles
    )
    
    embedder = OpenAIEmbedder(client)
    pipeline = EmbeddingPipeline(loader, embedder)
    
    ids, embeddings, payloads = pipeline.run()
    
    assert len(ids) == num_articles
    assert all(len(vec) == 3 for vec in embeddings)

@pytest.mark.parametrize("num_articles", [1, 3])
def test_pipeline_hf_mock_pipeline(setup_pipeline_json, num_articles):
    folder = setup_pipeline_json(num_articles)
    loader = JSONArticleLoader(folder.name)
    
    with patch("embeddings.infrastructure.hf_embedder.SentenceTransformer.encode") as mock_encode:
        mock_encode.return_value = np.array([[0.8, 0.9, 1.0]] * num_articles)
        embedder = HFEmbedder("all-MiniLM-L6-v2")
        pipeline = EmbeddingPipeline(loader, embedder)
        
        ids, embeddings, payloads = pipeline.run()
        
        assert len(ids) == num_articles
        assert all(len(vec) == 3 for vec in embeddings)
