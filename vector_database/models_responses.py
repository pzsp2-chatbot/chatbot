from typing import List

from pydantic import BaseModel


class PayloadResponse(BaseModel):
    title: str
    created: str
    modified: str
    language: str
    doi: str
    url: str
    authors: List[str]
    author_affiliations: List[str]
    abstract: str
    keywords: List[str]
    document_id: str


class ScoredPointResponse(BaseModel):
    score: float
    payload: PayloadResponse
