import hashlib
import math

DEFAULT_VECTOR_SIZE = 256


def embed(text: str, vector_size: int = DEFAULT_VECTOR_SIZE) -> list[float]:
    text = (text or "").lower().strip()
    vec = [0.0] * vector_size

    for token in text.split():
        h = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % vector_size
        sign = 1.0 if (h[4] % 2 == 0) else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]
