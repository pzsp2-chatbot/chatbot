import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from embeddings.infrastructure.dummy_embedder import DummyEmbedder
from embeddings.infrastructure.hf_embedder import HFEmbedder
from embeddings.infrastructure.openai_embedder import OpenAIEmbedder
from openai import OpenAI

def test_dummy_embedder_output():
    texts = ["Hello world", "Test embedding"]
    embedder = DummyEmbedder(vector_size=3)
    embeddings = embedder.embed(texts)
    assert all(len(vec) == 3 for vec in embeddings)
    print("Dummy embeddings:", embeddings)

def test_openai_embedder_mock():
    texts = ["Hello world", "Test embedding"]

    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in texts]
    )

    embedder = OpenAIEmbedder(client)
    embeddings = embedder.embed(texts)
    assert embeddings == [[0.1, 0.2, 0.3]] * len(texts)
    print("Mock OpenAI embeddings:", embeddings)

@pytest.mark.parametrize("num_articles", [1, 3])
def test_hf_embedder_mock(num_articles):
    texts = [f"HF Test {i}" for i in range(num_articles)]
    with patch("embeddings.infrastructure.hf_embedder.SentenceTransformer.encode") as mock_encode:
        mock_encode.return_value = np.array([[0.5, 0.6, 0.7]] * num_articles)
        embedder = HFEmbedder("all-MiniLM-L6-v2")
        embeddings = embedder.embed(texts)
        assert embeddings == [[0.5, 0.6, 0.7]] * num_articles
        print(f"HF mock embeddings ({num_articles} articles):", embeddings)
