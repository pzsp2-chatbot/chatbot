import argparse
import json
from dataclasses import dataclass
from typing import Any

import requests

from embeddings.service import embed


@dataclass(frozen=True)
class SearchConfig:
    api: str
    collection: str
    query: str
    vector_size: int
    top_k: int
    filter_json: str | None


class SearchCli:
    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        vector = embed(self.cfg.query, self.cfg.vector_size)
        payload = self._build_payload(vector)
        res = self._post_json(
            f"{self.cfg.api}/collections/{self.cfg.collection}/search", payload
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    def _build_payload(self, vector: list[float]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vector": vector,
            "top_k": self.cfg.top_k,
            "filter": {},
        }

        if self.cfg.filter_json:
            payload["filter"] = json.loads(self.cfg.filter_json)

        return payload

    @staticmethod
    def _post_json(
        url: str, payload: dict[str, Any], timeout: int = 30
    ) -> dict[str, Any]:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()


def parse_args() -> SearchConfig:
    p = argparse.ArgumentParser(description="Search: text -> embed -> /search")
    p.add_argument(
        "--api", required=True, help="API base URL, e.g. http://localhost:8001"
    )
    p.add_argument("--collection", required=True, help="Collection name, e.g. articles")
    p.add_argument("--query", required=True, help="Text query to search")
    p.add_argument(
        "--vector-size",
        type=int,
        default=256,
        help="Embedding vector size (must match collection)",
    )
    p.add_argument("--top-k", type=int, default=3, help="How many results to return")
    p.add_argument("--filter", default=None, help="Optional JSON filter dict (string)")
    a = p.parse_args()

    return SearchConfig(
        api=a.api,
        collection=a.collection,
        query=a.query,
        vector_size=a.vector_size,
        top_k=a.top_k,
        filter_json=a.filter,
    )


def main() -> None:
    cfg = parse_args()
    SearchCli(cfg).run()


if __name__ == "__main__":
    main()
