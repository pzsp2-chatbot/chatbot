import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from embeddings.service import embed


class IngestError(Exception):
    pass


class CollectionCreateError(IngestError):
    pass


class InsertItemError(IngestError):
    pass


@dataclass(frozen=True)
class IngestConfig:
    api: str
    collection: str
    vector_size: int
    data_dir: Path
    limit: int


class IngestClient:
    def __init__(self, cfg: IngestConfig) -> None:
        self.cfg = cfg

    def run(self) -> int:
        try:
            self._ensure_collection()
            files = self._list_json_files()
            inserted = self._ingest_files(files)
            print(
                f"Inserted {inserted} documents into collection '{self.cfg.collection}'"
            )
            return 0
        except IngestError as e:
            print(f"[INGEST ERROR] {e}")
            return 1
        except requests.RequestException as e:
            print(f"[HTTP ERROR] {e}")
            return 1
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"[DATA ERROR] {e}")
            return 1
        except Exception as e:
            print(f"[UNEXPECTED ERROR] {type(e).__name__}: {e}")
            return 1

    def _ensure_collection(self) -> None:
        r = requests.post(
            f"{self.cfg.api}/collections",
            json={"name": self.cfg.collection, "vector_size": self.cfg.vector_size},
            timeout=30,
        )
        if r.status_code in (200, 201):
            return
        if r.status_code == 409:
            return
        raise CollectionCreateError(
            f"Collection create failed: {r.status_code} {r.text}"
        )

    def _list_json_files(self) -> list[Path]:
        files = sorted(self.cfg.data_dir.glob("*.json"))
        if not files:
            raise IngestError(f"No .json files found in {self.cfg.data_dir}")
        return files

    def _ingest_files(self, files: list[Path]) -> int:
        inserted = 0
        for fp in files:
            if self.cfg.limit and inserted >= self.cfg.limit:
                break

            doc = self._load_json(fp)
            item = self._build_item(doc, fp)

            if item is None:
                continue
            self._send_item(item)
            inserted += 1

        return inserted

    def _load_json(self, fp: Path) -> dict[str, Any]:
        return json.loads(fp.read_text(encoding="utf-8"))

    def _build_item(self, doc: dict[str, Any], fp: Path) -> dict[str, Any] | None:
        title = self._to_str(doc.get("title")).strip()
        language = self._to_str(doc.get("language")).strip()
        abstract = self._to_str(doc.get("abstract")).strip()
        doi = self._to_str(doc.get("doi")).strip()
        url = self._to_str(doc.get("url")).strip()

        created = self._to_yyyy_mm_dd(doc.get("created"))
        modified = self._to_yyyy_mm_dd(doc.get("modified"))

        if not title or not language or not abstract or not doi or not url:
            print(f"Skipping {fp.name}: missing mandatory text fields")
            return None

        author_names, author_affils = self._parse_authors(
            doc.get("authors", []) or [], fp
        )
        if not author_names:
            print(f"Skipping {fp.name}: no valid authors")
            return None

        keywords = self._parse_keywords(doc.get("keywords"))
        if not keywords:
            keywords = self._keywords_from_title(title)
        if not keywords:
            keywords = ["general"]

        text_for_vec = f"{title}\n\n{abstract}"
        vector = embed(text_for_vec, vector_size=self.cfg.vector_size)

        payload = {
            "title": title,
            "language": language,
            "created": created,
            "modified": modified,
            "doi": doi,
            "url": url,
            "abstract": abstract,
            "authors": author_names,
            "author_affiliations": author_affils,
            "keywords": keywords,
        }

        return {"vector": vector, "payload": payload}

    def _send_item(self, item: dict[str, Any]) -> None:
        r = requests.post(
            f"{self.cfg.api}/collections/{self.cfg.collection}/items",
            json=item,
            timeout=60,
        )
        if r.status_code in (200, 201):
            return
        raise InsertItemError(f"Insert failed: {r.status_code} {r.text}")

    @staticmethod
    def _to_yyyy_mm_dd(value: Any) -> str:
        if not value:
            return ""
        return str(value)[:10]

    @staticmethod
    def _to_str(value: Any) -> str:
        return "" if value is None else str(value)

    def _parse_authors(
        self, authors: list[dict[str, Any]], fp: Path
    ) -> tuple[list[str], list[str]]:
        names: list[str] = []
        affils: list[str] = []

        for a in authors:
            name = (a.get("full_name") or "").strip()
            aff = (a.get("affiliation") or "").strip()

            if not name:
                continue
            if not aff:
                aff = "unknown"

            names.append(name)
            affils.append(aff)

        return names, affils

    @staticmethod
    def _parse_keywords(kw: Any) -> list[str]:
        if kw is None:
            return []
        if isinstance(kw, str):
            return [k.strip() for k in kw.split(",") if k.strip()]
        if isinstance(kw, list):
            return [str(k).strip() for k in kw if str(k).strip()]
        return []

    @staticmethod
    def _keywords_from_title(title: str, max_k: int = 6) -> list[str]:
        stop = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "on",
            "for",
            "with",
            "through",
            "between",
            "at",
            "by",
            "from",
            "into",
            "as",
            "is",
            "are",
            "was",
            "were",
            "study",
            "studying",
            "analysis",
            "using",
            "based",
            "via",
        }
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", title.lower())
        kws: list[str] = []
        for w in words:
            if len(w) < 3 or w in stop:
                continue
            if w not in kws:
                kws.append(w)
            if len(kws) >= max_k:
                break
        return kws


def parse_args() -> IngestConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8001")
    p.add_argument("--collection", default="articles")
    p.add_argument("--vector-size", type=int, default=256)
    p.add_argument("--data-dir", required=True, help="folder with .json files")
    p.add_argument(
        "--limit", type=int, default=50, help="number of successfully inserted docs"
    )
    a = p.parse_args()

    return IngestConfig(
        api=a.api,
        collection=a.collection,
        vector_size=a.vector_size,
        data_dir=Path(a.data_dir),
        limit=a.limit,
    )


def main() -> None:
    cfg = parse_args()
    client = IngestClient(cfg)
    raise SystemExit(client.run())


if __name__ == "__main__":
    main()
