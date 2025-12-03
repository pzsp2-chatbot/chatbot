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
                data = json.loads(file.read_text("utf-8"))
                authors = [Author(**a) for a in data.get("authors", [])]
                articles.append(
                    Article(
                        id=data["id"],
                        title=data["title"],
                        created=data["created"],
                        modified=data["modified"],
                        doi=data.get("doi"),
                        url=data.get("url"),
                        language=data["language"],
                        authors=authors,
                        abstract_pl=data.get("abstract_pl"),
                        abstract_en=data.get("abstract_en"),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ValueError(f"Failed to load article from {file.name}: {e}")
        return articles
