from embeddings.interfaces.embedder import IEmbedder
from typing import List
from sentence_transformers import SentenceTransformer

class HFEmbedder(IEmbedder):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model '{model_name}': {e}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            clean_texts = [t if t is not None else "" for t in texts]
            return self.model.encode(clean_texts).tolist()
        except Exception as e:
            raise RuntimeError(f"HuggingFace embedding failed: {e}")
