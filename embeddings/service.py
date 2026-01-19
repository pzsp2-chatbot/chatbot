import os
from functools import lru_cache
from typing import List, Optional

DEFAULT_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(DEFAULT_MODEL_NAME)

def embed(text: str, vector_size: Optional[int] = None) -> List[float]:
    text = (text or "").strip()
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)

    vec = vec.tolist()
    if vector_size is not None and vector_size != len(vec):
        raise ValueError(
            f"vector_size mismatch: got {vector_size}, but model returns {len(vec)}. "
            f"Use --vector-size {len(vec)} or recreate collection with that size."
        )
    return vec