from embeddings.pipeline import EmbeddingPipeline
from embeddings.infrastructure.json_loader import JSONArticleLoader
from embeddings.infrastructure.dummy_embedder import DummyEmbedder
from embeddings.infrastructure.hf_embedder import HFEmbedder
from embeddings.infrastructure.openai_embedder import OpenAIEmbedder
from embeddings.config import settings
from openai import OpenAI

EMBEDDER_TYPE = "dummy"  # "dummy"/"hf"/"openai"

def main():
    loader = JSONArticleLoader(settings.JSON_FOLDER)

    if EMBEDDER_TYPE == "dummy":
        embedder = DummyEmbedder(vector_size=768)
    elif EMBEDDER_TYPE == "hf":
        embedder = HFEmbedder(model_name="all-MiniLM-L6-v2")
    elif EMBEDDER_TYPE == "openai":
        client = OpenAI(api_key=settings.OPENAI_KEY)
        embedder = OpenAIEmbedder(client)
    else:
        raise ValueError(f"Unknown embedder type: {EMBEDDER_TYPE}")

    pipeline = EmbeddingPipeline(loader, embedder)
    try:
        ids, embeddings, payloads = pipeline.run()
        print(f"Generated embeddings for {len(ids)} articles using {EMBEDDER_TYPE} embedder.")
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
