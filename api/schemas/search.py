from pydantic import BaseModel


class SearchResult(BaseModel):
    text: str
    title: str
    source: str
    doc_id : str
    score : float


class SearchResponse(BaseModel):
    query : str
    results: list[SearchResult]
    answer: str
