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
    abstract_pl: Optional[str]
    abstract_en: Optional[str]

    def to_text(self) -> str:
        authors = ", ".join(a.full_name for a in self.authors)
        affiliations = ", ".join(a.affiliation or "" for a in self.authors)
        parts = [
            f"Title: {self.title}",
            f"Authors: {authors}",
            f"Affiliations: {affiliations}",
            f"Language: {self.language}",
            f"DOI: {self.doi}",
        ]
        if self.abstract_en:
            parts.append(f"Abstract (EN): {self.abstract_en}")
        if self.abstract_pl:
            parts.append(f"Abstract (PL): {self.abstract_pl}")
        return "\n".join(parts)
