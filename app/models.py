from pydantic import BaseModel, Field
from typing import List, Optional

class IngestRequest(BaseModel):
    id: Optional[str] = None
    text: str

class IngestResponse(BaseModel):
    id: str
    chunks: int

class SearchItem(BaseModel):
    id: str
    doc_id: str
    score: float
    text: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchItem]

class QARequest(BaseModel):
    question: str
    top_k: int = 5

class QAResponse(BaseModel):
    question: str
    answer: str
    sources: List[SearchItem]

class CompletenessRequest(BaseModel):
    question: str
    top_k: int = 5
    threshold: float = 0.25

class CompletenessResponse(BaseModel):
    question: str
    complete: bool
    avg_score: float
    top_k: int
