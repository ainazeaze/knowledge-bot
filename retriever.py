"""RAG retriever: search knowledge base and generate grounded answers."""

import config
from ingestion import KnowledgeStore


def _make_llm_client():
    """Return (client, model, provider_name) based on config, or None if unavailable."""
    provider = config.LLM_PROVIDER.lower()

    if provider == "claude":
        if not config.ANTHROPIC_API_KEY:
            return None
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        return (client, "claude-sonnet-4-6", "Claude")

    if provider == "ollama":
        try:
            from openai import OpenAI
            client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
            return (client, config.OLLAMA_MODEL, f"Ollama ({config.OLLAMA_MODEL})")
        except ImportError:
            return None

    return None


class Retriever:
    """Retrieves relevant context and generates answers using an LLM."""

    def __init__(self, store: KnowledgeStore):
        self.store = store
        result = _make_llm_client()
        if result:
            self._client, self._model, self.provider_name = result
            self.llm_available = True
        else:
            self._client = self._model = None
            self.provider_name = "none"
            self.llm_available = False

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Pure vector search — no LLM. Returns ranked results."""
        return self.store.search(query, top_k)

    def ask(self, question: str) -> dict:
        """RAG pipeline: retrieve context, then generate an answer.

        Returns a dict with:
            - answer: str
            - sources: list[dict]
            - used_llm: bool
        """
        results = self.store.search(question)

        if not results:
            return {
                "answer": "I don't have anything saved that's relevant to that question. "
                          "Try saving some content first with `/save`.",
                "sources": [],
                "used_llm": False,
            }

        if not self.llm_available:
            return {
                "answer": self._format_search_results(results),
                "sources": results,
                "used_llm": False,
            }

        context = self._build_context(results)
        answer = self._generate_answer(question, context, results)

        return {
            "answer": answer,
            "sources": results,
            "used_llm": True,
        }

    def _build_context(self, results: list[dict]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            source_label = r["source"] if r["source"] != "manual" else "manual note"
            parts.append(f"[Source {i}: {r['title']} ({source_label})]\n{r['text']}\n")
        return "\n---\n".join(parts)

    def _generate_answer(self, question: str, context: str, results: list[dict]) -> str:
        system_prompt = (
            "You are a helpful knowledge base assistant. Answer the user's question "
            "based ONLY on the provided context. If the context doesn't contain enough "
            "information to fully answer, say so honestly. Reference which sources you "
            "used. Keep answers concise but complete."
        )
        user_message = f"Context from my knowledge base:\n\n{context}\n\n---\n\nQuestion: {question}"

        try:
            if config.LLM_PROVIDER.lower() == "claude":
                return self._call_claude(system_prompt, user_message)
            else:
                return self._call_openai_compat(system_prompt, user_message)
        except Exception as e:
            return (
                f"LLM error: {e}\n\nHere are the raw search results instead:\n\n"
                + self._format_search_results(results)
            )

    def _call_claude(self, system_prompt: str, user_message: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _call_openai_compat(self, system_prompt: str, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def _format_search_results(self, results: list[dict]) -> str:
        if not results:
            return "No results found."

        lines = ["Here's what I found:\n"]
        for i, r in enumerate(results, 1):
            score_pct = f"{r['score']:.0%}"
            source = r["source"] if r["source"] != "manual" else "manual note"
            snippet = r["text"][:300].replace("\n", " ").strip()
            if len(r["text"]) > 300:
                snippet += "..."
            lines.append(f"**{i}. {r['title']}** ({score_pct} match)")
            lines.append(f"   Source: {source}")
            lines.append(f"   {snippet}\n")

        return "\n".join(lines)
