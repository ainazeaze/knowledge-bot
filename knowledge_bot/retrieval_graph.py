from typing import TypedDict
from . import config
from knowledge_bot.ingestion import KnowledgeStore
from langchain_groq import ChatGroq
from pydantic import SecretStr
from langchain_core.messages import HumanMessage, SystemMessage

store = KnowledgeStore()
llm = ChatGroq(api_key=SecretStr(config.GROQ_API_KEY), model=config.GROQ_MODEL)

class RetrievalState(TypedDict):
    query : str
    original_query: str
    results : list[dict]
    attempts : int
    should_rewrite : bool

def search_node(state : RetrievalState) -> dict :
     results = store.search(state["query"])
     return {
         "results" : results,
         "attempts" : state["attempts"] + 1
     }

def grade_node(state : RetrievalState) -> dict:
    response = llm.invoke([SystemMessage(content="You are a relevance grader. Answer only YES or NO"),
        HumanMessage(content=f"Query: {state['original_query']}\n\nResult: {state['results'][0]['text']}")])
    return {
        "should_rewrite": "YES" not in response.content
    }
