from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_store
from api.schemas.documents import DocumentItem, DocumentListResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str, store=Depends(get_store)):  # noqa: B008
    deleted = store.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")


@router.get("", response_model=DocumentListResponse)
def get_documents(limit: int = 10, store=Depends(get_store)): # noqa: B008
    docs = store.list_documents(limit=limit)

    if not docs:
        raise HTTPException(status_code=404, detail="No document found")

    docs = [
        DocumentItem(
            doc_id=doc["doc_id"],
            title=doc["title"],
            source=doc["source"],
            added_at=doc["added_at"],
            total_chunks=doc["total_chunks"],
        )
        for doc in docs
    ]
    return DocumentListResponse(documents=docs)
