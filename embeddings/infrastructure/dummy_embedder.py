from embeddings.interfaces.embedder import IEmbedder
from typing import List
import random

class DummyEmbedder(IEmbedder):
    def __init__(self, vector_size: int = 768):
        self.vector_size = vector_size

    def embed(self, texts: List[str]):
        return [
            [random.random() for _ in range(self.vector_size)]
            for _ in texts
        ]
