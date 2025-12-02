from typing import Annotated, Optional, ClassVar, TypedDict, List
from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    VECTOR_MAX_SIZE: ClassVar[int] = 1024

    name: str = Field(..., min_length=1, max_length=64, description="Name of the collection")
    vector_size: int = Field(VECTOR_MAX_SIZE, ge=1, le=VECTOR_MAX_SIZE,
        description="Size of vectors in the collection")


class PayloadDict(TypedDict):
    title: str
    created: str
    modified: str
    language: str
    doi: str
    url: str
    authors: list[str]
    author_affiliations: list[str]
    abstract: str
    keywords: list[str]


class AddItemRequest(BaseModel):
    VECTOR_MAX_SIZE: ClassVar[int] = 1024

    vector: Annotated[list[float], Field(min_length=1, max_length=VECTOR_MAX_SIZE,
            description=f"Vector containing up to {VECTOR_MAX_SIZE} float values")]

    payload: PayloadDict = Field(..., description="Metadata describing an element")


class SearchFilterDict(TypedDict, total=False):
    title: Optional[str]
    starting_creation_date: Optional[str]
    ending_creation_date: Optional[str]
    starting_modification_date: Optional[str]
    ending_modification_date: Optional[str]
    language: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    authors: Optional[List[str]]
    author_affiliations: Optional[List[str]]
    abstract: Optional[str]
    keywords: Optional[List[str]]


class SearchItemRequest(BaseModel):
    VECTOR_MAX_SIZE: ClassVar[int] = 1024
    RESULTS_BY_DEFAULT: ClassVar[int] = 1

    vector: Annotated[list[float], Field(min_length=1, max_length=VECTOR_MAX_SIZE,
            description=f"Vector containing up to {VECTOR_MAX_SIZE} float values")]

    top_k: int = Field(RESULTS_BY_DEFAULT, gt=0, description=f"Number of results ({RESULTS_BY_DEFAULT} by default)")

    filter: Optional[SearchFilterDict] = Field(None,
        description="Optional metadata filter")
