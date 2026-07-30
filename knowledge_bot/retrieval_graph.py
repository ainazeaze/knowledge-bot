import time
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr

from . import config
from .logger import logger
from .store import get_store

llm = ChatGroq(api_key=SecretStr(config.GROQ_API_KEY), model=config.GROQ_MODEL)

class RetrievalState(TypedDict):
    query : str
    original_query: str
    results : list[dict]
    attempts : int
    should_rewrite : bool
    top_k : int

def search_node(state: RetrievalState) -> dict:
    t0 = time.perf_counter()
    results = get_store().search(state["query"], state["top_k"])
    elapsed = time.perf_counter() - t0
    top_score = results[0]["score"] if results else None
    logger.info(
        "search | attempt=%d query=%r results=%d top_score=%s latency=%.2fs",
        state["attempts"] + 1, state["query"], len(results), top_score, elapsed,
    )
    return {"results": results, "attempts": state["attempts"] + 1}

def grade_node(state: RetrievalState) -> dict:
    response = llm.invoke([
        SystemMessage(content="You are a relevance grader. Answer only YES or NO"),
        HumanMessage(content=f"Query: {state['original_query']}\n\nResult: {state['results'][0]['text']}"),
    ])
    should_rewrite = "YES" not in response.content
    logger.info("grade | relevant=%s query=%r", not should_rewrite, state["original_query"])
    return {"should_rewrite": should_rewrite}

def rewrite_node(state: RetrievalState) -> dict:
    response = llm.invoke([
        SystemMessage(content="You are a query rewriter. Answer with ONLY a better query"),
        HumanMessage(content=f"Rewrite this query to find more relevant results: {state['original_query']}"),
    ])
    new_query = response.content.strip()  # type: ignore[union-attr]
    logger.info("rewrite | original=%r rewritten=%r", state["original_query"], new_query)
    return {"query": new_query}

def should_continue(state: RetrievalState) -> str:
    if state["should_rewrite"] and state["attempts"] < 2:
        return "rewrite"
    return END



graph = StateGraph(RetrievalState)

graph.add_node("search", search_node)
graph.add_node("grade", grade_node)
graph.add_node("rewrite", rewrite_node)

graph.add_edge(START, "search")
graph.add_edge("search", "grade")
graph.add_edge("rewrite", "search")

graph.add_conditional_edges("grade", should_continue)

app = graph.compile()
