from typing import List
import numpy as np
import gensim.downloader as api
from embeddings.interfaces.embedder import IEmbedder

class GensimEmbedder(IEmbedder):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = api.load(model_name)
        self.dim = self.model.vector_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            words = [w for w in text.split() if w in self.model]
            if not words:
                vectors.append(np.zeros(self.dim).tolist())
            else:
                word_vecs = [self.model[w] for w in words]
                vectors.append(np.mean(word_vecs, axis=0).tolist())
        return vectors
