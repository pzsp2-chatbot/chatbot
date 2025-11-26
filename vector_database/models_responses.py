from pydantic import BaseModel


class PayloadResponse(BaseModel):
    text: str
    author: str
    published_at: str
    document_id: str


class ScoredPointResponse(BaseModel):
    score: float
    payload: PayloadResponse
