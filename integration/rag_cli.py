import argparse
import json
from typing import Any, Dict, List

import requests

from embeddings.service import embed
from integration.llm_client import llm_chat


def post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_fragments(items: List[Dict[str, Any]], top_k: int) -> List[str]:
    frags: List[str] = []
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


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG CLI: question -> embed -> search -> LLM answer")
    parser.add_argument("--api", required=True, help="http://localhost:8001")
    parser.add_argument("--collection", required=True, help="e.g. articles")
    parser.add_argument("--vector-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    vec = embed(args.question, args.vector_size)

    search_resp = post_json(
        f"{args.api}/collections/{args.collection}/search",
        {"vector": vec, "top_k": args.top_k, "filter": {}},
    )

    items = search_resp.get("items", [])
    fragments = extract_fragments(items, args.top_k)
    context = "\n\n---\n\n".join(fragments)

    messages = [
        {
            "role": "system",
            "content": "Answer using the provided context. If context is insufficient, say you don't know.",
        },
        {"role": "user", "content": f"QUESTION:\n{args.question}\n\nCONTEXT:\n{context}"},
    ]

    answer = llm_chat(messages)

    print(json.dumps({"question": args.question, "answer": answer, "context": fragments}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())