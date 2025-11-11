from typing import Dict, Any, Annotated, Optional, ClassVar
from pydantic import BaseModel, Field, field_validator


class CreateCollectionRequest(BaseModel):
    VECTOR_MAX_SIZE: ClassVar[int] = 1024

    name: str = Field(..., min_length=1, max_length=64, description="Name of the collection")
    vector_size: int = Field(VECTOR_MAX_SIZE, ge=1, le=VECTOR_MAX_SIZE,
        description="Size of vectors in the collection")


class AddItemRequest(BaseModel):
    VECTOR_MAX_SIZE: ClassVar[int] = 1024

    vector: Annotated[list[float], Field(min_length=1, max_length=VECTOR_MAX_SIZE,
            description=f"Vector containing up to {VECTOR_MAX_SIZE} float values")]

    payload: Dict[str, Any] = Field(..., description="Metadata describing an element")

    @field_validator("payload")
    def validate_payload_not_empty(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            raise ValueError("Payload cannot be empty")
        return value


class SearchItemRequest(BaseModel):
    VECTOR_MAX_SIZE: ClassVar[int] = 1024
    RESULTS_BY_DEFAULT: ClassVar[int] = 10
    FILTER_BY_DEFAULT: ClassVar[str] = None

    vector: Annotated[list[float], Field(min_length=1, max_length=VECTOR_MAX_SIZE,
            description=f"Vector containing up to {VECTOR_MAX_SIZE} float values")]

    top_k: int = Field(RESULTS_BY_DEFAULT, gt=0, description=f"Number of results ({RESULTS_BY_DEFAULT} by default)")

    filter: Optional[Dict[str, Any]] = Field(FILTER_BY_DEFAULT,
        description=f"Optional metadata filter (date of publication, author), {FILTER_BY_DEFAULT} by default")
