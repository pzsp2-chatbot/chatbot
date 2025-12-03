import pytest
from embeddings.interfaces.embedder import IEmbedder
from embeddings.infrastructure.st_embedder import STEmbedder
from embeddings.infrastructure.gensim_embedder import GensimEmbedder

SAMPLE_TEXTS = ["Ala ma kota", "Kot ma Alę"]


def assert_embedding_shape(embeddings, min_dim=128):
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(len(vec) >= min_dim for vec in embeddings)
    assert all(isinstance(x, float) for vec in embeddings for x in vec)


@pytest.mark.parametrize(
    "model_name, min_dim",
    [
        ("sentence-transformers/all-MiniLM-L6-v2", 256),
        ("sentence-transformers/LaBSE", 768),
    ],
)
def test_st_embedder_basic(model_name, min_dim):
    embedder = STEmbedder(model_name)
    embeddings = embedder.embed(SAMPLE_TEXTS)
    assert_embedding_shape(embeddings, min_dim=min_dim)


@pytest.mark.parametrize(
    "model_name, min_dim",
    [
        ("fasttext-wiki-news-subwords-300", 300),
        ("glove-wiki-gigaword-300", 300),
    ],
)
def test_gensim_embedder_basic(model_name, min_dim):
    embedder = GensimEmbedder(model_name)
    embeddings = embedder.embed(SAMPLE_TEXTS)
    assert_embedding_shape(embeddings, min_dim=min_dim)


def test_st_embedder_is_instance():
    embedder = STEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(embedder, IEmbedder)


def test_gensim_embedder_is_instance():
    embedder = GensimEmbedder("glove-wiki-gigaword-300")
    assert isinstance(embedder, IEmbedder)
