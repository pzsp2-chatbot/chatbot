import argparse
import json
from typing import Any, Dict

import requests

from embeddings.service import embed


def post_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search in Qdrant via vector_api: text -> embed -> /search"
    )
    parser.add_argument(
        "--api", required=True, help="API base URL, e.g. http://localhost:8001"
    )
    parser.add_argument(
        "--collection", required=True, help="Collection name, e.g. articles"
    )
    parser.add_argument("--query", required=True, help="Text query to search")
    parser.add_argument(
        "--vector-size",
        type=int,
        default=256,
        help="Embedding vector size (must match collection)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="How many results to return"
    )

    # filter вы можете передать как JSON строку, например:
    # --filter '{"language":"en","authors":["John Doe"]}'
    parser.add_argument(
        "--filter", default=None, help="Optional JSON filter dict (string)"
    )

    args = parser.parse_args()

    vector = embed(args.query, args.vector_size)

    payload: Dict[str, Any] = {
        "vector": vector,
        "top_k": args.top_k,
        "filter": {},
    }
    if args.filter:
        payload["filter"] = json.loads(args.filter)

    res = post_json(f"{args.api}/collections/{args.collection}/search", payload)

    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
