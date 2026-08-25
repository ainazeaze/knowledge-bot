from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_store
from api.schemas.search import SearchResponse, SearchResult
from knowledge_bot.retrieval_graph import app

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search(q: str, top_k: int = 5, store=Depends(get_store)):  # noqa : B008
    final_state = app.invoke(
        {
            "query": q,
            "original_query": q,
            "results": [],
            "attempts": 0,
            "should_rewrite": False,
            "top_k": top_k,
        }
    )
    results = final_state["results"]

    if not results:
        raise HTTPException(status_code=404, detail="No results found for that query")

    search_results = [
        SearchResult(
            text=result["text"],
            title=result["title"],
            source=result["source"],
            doc_id=result["doc_id"],
            score=result["score"],
        )
        for result in results
    ]

    return SearchResponse(query=q, results=search_results)
