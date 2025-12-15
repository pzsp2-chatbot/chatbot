import argparse
import json
import re
from pathlib import Path

import requests

from embeddings.service import embed


def to_yyyy_mm_dd(value) -> str:
    if not value:
        return ""
    return str(value)[:10]


def to_str(value) -> str:
    return "" if value is None else str(value)


def keywords_from_title(title: str, max_k: int = 6) -> list[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "through",
        "between", "at", "by", "from", "into", "as", "is", "are", "was", "were",
        "study", "studying", "analysis", "using", "based", "via",
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


def ensure_collection(api: str, name: str, vector_size: int):
    r = requests.post(f"{api}/collections", json={"name": name, "vector_size": vector_size})
    if r.status_code not in (200, 201):
        print("Collection create:", r.status_code, r.text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8001")
    p.add_argument("--collection", default="articles")
    p.add_argument("--vector-size", type=int, default=256)
    p.add_argument("--data-dir", required=True, help="folder with .json files")
    p.add_argument("--limit", type=int, default=50, help="number of successfully inserted docs")
    args = p.parse_args()

    ensure_collection(args.api, args.collection, args.vector_size)

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No .json files found in {data_dir}")

    inserted = 0

    for fp in files:
        if args.limit and inserted >= args.limit:
            break

        doc = json.loads(fp.read_text(encoding="utf-8"))

        title = to_str(doc.get("title")).strip()
        language = to_str(doc.get("language")).strip()
        abstract = to_str(doc.get("abstract")).strip()
        doi = to_str(doc.get("doi")).strip()
        url = to_str(doc.get("url")).strip()

        created = to_yyyy_mm_dd(doc.get("created"))
        modified = to_yyyy_mm_dd(doc.get("modified"))

        if not title or not language or not abstract or not doi or not url:
            print(f"Skipping {fp.name}: missing mandatory text fields")
            continue

        authors = doc.get("authors", []) or []
        author_names: list[str] = []
        author_affils: list[str] = []

        for a in authors:
            name = (a.get("full_name") or "").strip()
            aff = (a.get("affiliation") or "").strip()

            if not name:
                continue
            if not aff:
                aff = "unknown"

            author_names.append(name)
            author_affils.append(aff)

        if not author_names:
            print(f"Skipping {fp.name}: no valid authors")
            continue

        kw = doc.get("keywords", [])
        if kw is None:
            keywords = []
        elif isinstance(kw, str):
            keywords = [k.strip() for k in kw.split(",") if k.strip()]
        elif isinstance(kw, list):
            keywords = [str(k).strip() for k in kw if str(k).strip()]
        else:
            keywords = []

        if not keywords:
            keywords = keywords_from_title(title)
        if not keywords:
            keywords = ["general"]

        text_for_vec = f"{title}\n\n{abstract}"
        vector = embed(text_for_vec, vector_size=args.vector_size)

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

        item = {"id": doc.get("id", fp.stem), "vector": vector, "payload": payload}

        r = requests.post(f"{args.api}/collections/{args.collection}/items", json=item)
        if r.status_code not in (200, 201):
            print("FAILED:", r.status_code)
            print("Response:", r.text)
            print("Example payload:", payload)
            return

        inserted += 1

    print(f"Inserted {inserted} documents into collection '{args.collection}'")


if __name__ == "__main__":
    main()