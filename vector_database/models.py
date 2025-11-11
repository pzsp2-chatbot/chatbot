from typing import Dict, Any, Annotated, Optional, ClassVar
from pydantic import BaseModel, Field, field_validator


class CreateCollectionRequest(BaseModel):
    VECTOR_SIZE: ClassVar[int] = 1024

    name: str = Field(..., min_length=1, max_length=64, description="Name of the collection")
    vector_size: int = Field(VECTOR_SIZE, ge=1024, le=1024, description="Size of vectors in the collection")


class AddItemRequest(BaseModel):
    VECTOR_SIZE: ClassVar[int] = 1024

    id: int = Field(..., description="Id of an element")

    vector: Annotated[list[float], Field(min_length=VECTOR_SIZE, max_length=VECTOR_SIZE,
            description=f"Vector containing exactly {VECTOR_SIZE} float values")]

    metadata: Dict[str, Any] = Field(..., description="Metadata describing an element")

    @field_validator("metadata")
    def validate_metadata_not_empty(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            raise ValueError("Metadata cannot be empty")
        return value


class SearchItemRequest(BaseModel):
    VECTOR_SIZE: ClassVar[int] = 1024
    RESULTS_BY_DEFAULT: ClassVar[int] = 10
    FILTER_BY_DEFAULT: ClassVar[str] = None

    vector: Annotated[list[float], Field(min_length=1024, max_length=1024,
            description=f"Vector containing exactly {VECTOR_SIZE} float values")]

    top_k: int = Field(RESULTS_BY_DEFAULT, gt=0, description=f"Number of results ({RESULTS_BY_DEFAULT} by default)")

    filter: Optional[Dict[str, Any]] = Field(FILTER_BY_DEFAULT,
        description=f"Optional metadata filter (date of publication, author), {FILTER_BY_DEFAULT} by default")
