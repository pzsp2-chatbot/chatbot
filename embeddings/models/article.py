from dataclasses import dataclass
from typing import List, Optional
from .author import Author


@dataclass
class Article:
    id: str
    title: str
    language: str
    created: str
    modified: str
    doi: Optional[str]
    url: Optional[str]
    authors: List[Author]
    abstract: Optional[str]
    keywords: List[str]

    def to_text(self) -> str:
        authors = ", ".join(a.full_name for a in self.authors)
        affiliations = ", ".join(
            a.affiliation for a in self.authors if a.affiliation
        )

        parts = [
            f"Title: {self.title}",
            f"Authors: {authors}",
        ]

        if affiliations:
            parts.append(f"Affiliations: {affiliations}")

        parts.append(f"Language: {self.language}")

        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")

        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")

        return "\n".join(parts)
