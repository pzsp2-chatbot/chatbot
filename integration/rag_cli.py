import argparse
import json
from dataclasses import dataclass
from typing import Any

import requests

from embeddings.service import embed
from integration.llm_client import llm_chat


@dataclass(frozen=True)
class RagConfig:
    api: str
    collection: str
    vector_size: int
    top_k: int
    question: str


class RagCli:
    def __init__(self, cfg: RagConfig) -> None:
        self.cfg = cfg

    def run(self) -> dict[str, Any]:
        vec = embed(self.cfg.question, self.cfg.vector_size)

        search_resp = self._post_json(
            f"{self.cfg.api}/collections/{self.cfg.collection}/search",
            {"vector": vec, "top_k": self.cfg.top_k, "filter": {}},
        )

        items = search_resp.get("items", [])
        fragments = self._extract_fragments(items, self.cfg.top_k)
        context = "\n\n---\n\n".join(fragments)

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer using the provided context. "
                    "If context is insufficient, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"QUESTION:\n{self.cfg.question}\n\nCONTEXT:\n{context}",
            },
        ]

        answer = llm_chat(messages)

        return {"question": self.cfg.question, "answer": answer, "context": fragments}

    @staticmethod
    def _post_json(
        url: str, payload: dict[str, Any], timeout: int = 60
    ) -> dict[str, Any]:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _extract_fragments(items: list[dict[str, Any]], top_k: int) -> list[str]:
        frags: list[str] = []
        for it in items:
            payload = it.get("payload", {}) if isinstance(it, dict) else {}
            title = payload.get("title") or ""
            abstract = payload.get("abstract") or ""
            text = (title + "\n" + abstract).strip()
            if text:
                frags.append(text)
            if len(frags) >= top_k:
                break
        return frags


def parse_args() -> RagConfig:
    p = argparse.ArgumentParser(
        description="RAG CLI: question -> embed -> search -> LLM answer"
    )
    p.add_argument("--api", required=True, help="http://localhost:8001")
    p.add_argument("--collection", required=True, help="e.g. articles")
    p.add_argument("--vector-size", type=int, default=256)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--question", required=True)
    a = p.parse_args()

    return RagConfig(
        api=a.api,
        collection=a.collection,
        vector_size=a.vector_size,
        top_k=a.top_k,
        question=a.question,
    )


def main() -> None:
    cfg = parse_args()
    result = RagCli(cfg).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
