from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr

from api.dependencies import get_store
from api.schemas.search import SearchResponse, SearchResult
from knowledge_bot import config
from knowledge_bot.retrieval_graph import app

router = APIRouter(prefix="/search", tags=["search"])

llm = ChatGroq(api_key=SecretStr(config.GROQ_API_KEY), model=config.GROQ_MODEL)


@router.get("", response_model=SearchResponse)
def search(q: str, top_k: int = 5, store=Depends(get_store)):  # noqa: B008
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

    context = "\n\n".join(r.text for r in search_results)
    response = llm.invoke([
        SystemMessage(content="Answer the question using only the provided context. Be concise."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {q}"),
    ])
    answer = response.content  # type: ignore[union-attr]

    return SearchResponse(query=q, answer=answer, results=search_results)
