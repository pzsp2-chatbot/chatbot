import pytest
from embeddings.pipeline import EmbeddingPipeline
from embeddings.infrastructure.embedder_factory import EmbedderFactory
from embeddings.interfaces.embedder import IEmbedder
from embeddings.models.article import Article
from embeddings.models.author import Author

class MockLoader:
    def load_all(self):
        return [
            Article(
                id="1",
                title="Test Article 1",
                language="pl",
                created="2025-01-01",
                modified="2025-01-01",
                doi=None,
                url=None,
                authors=[Author(full_name="Jan Kowalski", affiliation="Uniwersytet Warszawski")],
                abstract_pl="Streszczenie po polsku",
                abstract_en="Abstract in English"
            ),
            Article(
                id="2",
                title="Test Article 2",
                language="en",
                created="2025-01-02",
                modified="2025-01-02",
                doi=None,
                url=None,
                authors=[Author(full_name="Anna Nowak", affiliation="Uniwersytet Jagielloński")],
                abstract_pl=None,
                abstract_en="Second abstract in English"
            ),
        ]

def assert_embeddings(embeddings, min_dim=128):
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    for vec in embeddings:
        assert isinstance(vec, list)
        assert len(vec) >= min_dim
        assert all(isinstance(x, float) for x in vec)

EMBEDDERS = [
    ("st_minilm", 384),        
    ("st_labse", 768),         
    ("fasttext", 300),         
    ("glove", 300),            
]

@pytest.mark.parametrize("embedder_name, min_dim", EMBEDDERS)
def test_pipeline_with_factory(embedder_name, min_dim):
    loader = MockLoader()

    if embedder_name in ("st_labse", "fasttext", "glove"):
        pytest.skip(f"Skipping slow model {embedder_name} in CI")

    embedder = EmbedderFactory.create(embedder_name)
    assert isinstance(embedder, IEmbedder)

    pipeline = EmbeddingPipeline(loader, embedder)
    ids, embeddings, payloads = pipeline.run()

    assert ids == ["1", "2"]
    assert len(payloads) == 2
    assert_embeddings(embeddings, min_dim=min_dim)

def test_pipeline_no_articles_raises():
    class EmptyLoader:
        def load_all(self):
            return []

    embedder = EmbedderFactory.create("st_minilm")
    pipeline = EmbeddingPipeline(EmptyLoader(), embedder)

    with pytest.raises(RuntimeError, match="No articles to embed"):
        pipeline.run()
