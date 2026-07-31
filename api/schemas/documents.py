from pydantic import BaseModel


class DocumentItem(BaseModel):
    doc_id : str
    title: str
    source: str
    added_at: str
    total_chunks : int


class DocumentListResponse(BaseModel):
    documents : list[DocumentItem]
