from embeddings.pipeline import EmbeddingPipeline
from embeddings.infrastructure.json_loader import JSONArticleLoader
from embeddings.infrastructure.embedder_factory import EmbedderFactory
from embeddings.config import settings

EMBEDDER_TYPE = "st_minilm"


def main():
    loader = JSONArticleLoader(settings.JSON_FOLDER)
    factory = EmbedderFactory()
    embedder = factory.create(EMBEDDER_TYPE)

    pipeline = EmbeddingPipeline(loader, embedder)

    try:
        ids, embeddings, payloads = pipeline.run()
        print(f"Generated embeddings for {len(ids)} articles using '{EMBEDDER_TYPE}'.")
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")


if __name__ == "__main__":
    main()
