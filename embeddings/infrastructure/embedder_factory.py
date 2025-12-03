from embeddings.infrastructure.st_embedder import STEmbedder
from embeddings.infrastructure.gensim_embedder import GensimEmbedder
from embeddings.interfaces.embedder import IEmbedder


class EmbedderFactory:
    @staticmethod
    def create(name: str) -> IEmbedder:
        name = name.lower()
        if name == "st_minilm":
            return STEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        if name == "st_labse":
            return STEmbedder("sentence-transformers/LaBSE")
        if name == "fasttext":
            return GensimEmbedder("fasttext-wiki-news-subwords-300")
        if name == "glove":
            return GensimEmbedder("glove-wiki-gigaword-300")
        raise ValueError(f"Unknown embedder name: {name}")
