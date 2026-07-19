
import sys
from unittest.mock import MagicMock
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

from knowledge_bot import config
from knowledge_bot.ingestion import KnowledgeStore
from knowledge_bot.retriever import Retriever


# TODO : add more test samples
TEST_CASES = [
    {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a subset of AI where systems learn from data to make predictions or decisions without being explicitly programmed.",
    },
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "ground_truth": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Unsupervised learning finds patterns in data without labels.",
    },
]


def _build_judge_llm():
    from openai import OpenAI
    from ragas.llms import llm_factory

    # Ollama speaks OpenAI's API format, so we point the client at localhost.
    client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
    return llm_factory(config.OLLAMA_MODEL, client=client)


def _build_judge_embeddings():
    from ragas.embeddings import HuggingfaceEmbeddings

    # Reuse the same embedding model the bot uses — no extra download needed.
    return HuggingfaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def collect_samples(retriever: Retriever) -> list[dict]:
    """Run each test question through the bot and collect (question, contexts, answer)."""
    samples = []
    total = len(TEST_CASES)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"  [{i}/{total}] {case['question'][:60]}...")
        result = retriever.ask(case["question"], user_id="eval")
        contexts = [r["text"] for r in result["sources"]]

        samples.append({
            "user_input": case["question"],
            "retrieved_contexts": contexts,
            "response": result["answer"],
            "reference": case["ground_truth"],
        })

    return samples


def print_scorecard(results) -> None:
    df = results.to_pandas()

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available = [m for m in metric_cols if m in df.columns]

    print("RAGAS Scorecard")

    col_width = 20
    header = f"{'Question':<35}" + "".join(f"{m[:col_width]:>{col_width}}" for m in available)
    print(header)

    for _, row in df.iterrows():
        question = str(row["user_input"])[:33] + ".." if len(str(row["user_input"])) > 35 else str(row["user_input"])
        scores = "".join(
            f"{row[m]:>{col_width}.2f}" if row[m] == row[m] else f"{'N/A':>{col_width}}"
            for m in available
        )
        print(f"{question:<35}{scores}")

    print(f"{'MEAN':<35}", end="")
    for m in available:
        mean = df[m].mean()
        print(f"{mean:>{col_width}.2f}", end="")
    print("\n")

    print("Legend:")
    print("  faithfulness      — answer only uses info from retrieved context (no hallucination)")
    print("  answer_relevancy  — answer actually addresses the question asked")
    print("  context_precision — retrieved chunks were relevant (not noisy)")
    print("  context_recall    — retrieved chunks contained enough to answer fully")


def main():
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    except ImportError:
        print("RAGAS not installed. Run: pip install ragas")
        return

    print("Step 1/3 — Running test questions through the bot...")
    store = KnowledgeStore()
    retriever = Retriever(store)
    samples = collect_samples(retriever)

    print("\nStep 2/3 — Building RAGAS dataset...")
    dataset = EvaluationDataset.from_list(samples)

    print("\nStep 3/3 — Scoring with LLM judge (this takes a moment)...")
    judge_llm = _build_judge_llm()

    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=judge_llm),
            AnswerRelevancy(llm=judge_llm),
            ContextPrecision(llm=judge_llm),
            ContextRecall(llm=judge_llm),
        ],
    )

    print_scorecard(results)


if __name__ == "__main__":
    main()
