import json
from pathlib import Path
from embeddings.models.article import Article
from embeddings.models.author import Author
from embeddings.interfaces.loader import IArticleLoader


class JSONArticleLoader(IArticleLoader):
    def __init__(self, folder: str):
        self.folder = folder

    def load_all(self):
        articles = []

        for file in Path(self.folder).glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))

                authors = [
                    Author(
                        full_name=a["full_name"],
                        affiliation=a.get("affiliation"),
                    )
                    for a in data.get("authors", [])
                ]

                raw_keywords = data.get("keywords") or ""

                keywords = [
                    k.strip()
                    for k in raw_keywords.split(";")
                    if k.strip()
                ]

                article = Article(
                    id=data["id"],
                    title=data["title"],
                    language=data["language"],
                    created=data["created"],
                    modified=data["modified"],
                    doi=data.get("doi"),
                    url=data.get("url"),
                    authors=authors,
                    abstract=data.get("abstract"),
                    keywords=keywords,
                )

                articles.append(article)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ValueError(
                    f"Failed to load article from {file.name}: {e}"
                ) from e

        return articles
