from typing import List
from sentence_transformers import SentenceTransformer
from embeddings.interfaces.embedder import IEmbedder

class STEmbedder(IEmbedder):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
