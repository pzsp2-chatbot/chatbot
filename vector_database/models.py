from typing import Dict, Any, Annotated, Optional
from pydantic import BaseModel, Field, field_validator


class CreateCollectionRequest(BaseModel):
    VECTOR_SIZE = 1024

    name: str = Field(..., description="Name of the collection")
    vector_size: int = Field(VECTOR_SIZE, gt=0, description="Size of vectors in the collection")


class AddItemRequest(BaseModel):
    id: int = Field(..., description="Id of an element")

    vector: Annotated[list[float], Field(min_length=1024, max_length=1024,
            description="Vector containing exactly 1024 float values")]

    metadata: Dict[str, Any] = Field(..., description="Metadata describing an element")

    @field_validator("metadata")
    def validate_metadata_not_empty(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            raise ValueError("Metadata cannot be empty")
        return value


class SearchItemRequest(BaseModel):
    RESULTS_BY_DEFAULT = 10
    FILTER_BY_DEFAULT = None

    vector: Annotated[list[float], Field(min_length=1024, max_length=1024,
            description="Vector containing exactly 1024 float values")]

    top_k: int = Field(RESULTS_BY_DEFAULT, gt=0, description="Number of results (10 by default)")

    filter: Optional[Dict[str, Any]] = Field(FILTER_BY_DEFAULT,
        description="Optional metadata filter (date of publication, author), None by default")
