"""
Step 6: Multi-document comparison.

The naive approach -- flatten every document's chunks into one index
and take the overall top-k -- breaks for comparison questions, because
the single best-matching document crowds out the others. If Hartford's
GL policy has a very strong match for "deductible," a flattened search
might return 4 Hartford chunks and zero from the other two carriers,
even though the question was "compare all three."

The fix: retrieve top-k *per document*, then combine before generating
an answer. This guarantees every indexed document gets a chance to
contribute to the answer instead of the retrieval step silently
dropping documents that were a slightly worse semantic match.
"""

from src.index import get_collection, get_embedder, list_indexed_documents, DEFAULT_COLLECTION_NAME
from src.llm import generate

COMPARISON_PROMPT_TEMPLATE = """You are comparing insurance policy documents to answer a question. \
Below is context retrieved separately from each document, labeled by carrier and coverage type.

Rules:
- Only use the context provided. Do not use outside knowledge.
- If a document's context does not address the question, say so explicitly for that document \
rather than omitting it or guessing.
- Cite carrier, coverage type, and page for every claim.
- If the question asks for a comparison, present the answer as a short table.

Context:
{context}

Question: {question}

Answer:"""


def retrieve_per_document(
    query: str, k_per_doc: int = 3, collection_name: str = DEFAULT_COLLECTION_NAME
) -> dict[str, list[dict]]:
    """Retrieve top-k chunks from EACH indexed document separately."""
    embedder = get_embedder()
    collection = get_collection(reset=False, collection_name=collection_name)
    query_embedding = embedder.encode([query]).tolist()

    results_by_doc: dict[str, list[dict]] = {}
    for doc in list_indexed_documents(collection_name=collection_name):
        doc_id = doc["doc_id"]
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=k_per_doc,
            where={"doc_id": doc_id},
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "text": results["documents"][0][i],
                    "page": results["metadatas"][0][i]["page"],
                    "carrier": results["metadatas"][0][i]["carrier"],
                    "coverage_type": results["metadatas"][0][i]["coverage_type"],
                    "distance": results["distances"][0][i],
                }
            )
        results_by_doc[doc_id] = hits
    return results_by_doc


def compare(question: str, k_per_doc: int = 3, collection_name: str = DEFAULT_COLLECTION_NAME) -> dict:
    per_doc_hits = retrieve_per_document(question, k_per_doc=k_per_doc, collection_name=collection_name)

    context_blocks = []
    for doc_id, hits in per_doc_hits.items():
        if not hits:
            continue
        carrier = hits[0]["carrier"]
        coverage = hits[0]["coverage_type"]
        block = f"--- Document: {doc_id} (carrier: {carrier}, coverage: {coverage}) ---\n"
        block += "\n".join(f"[page {h['page']}] {h['text']}" for h in hits)
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)
    prompt = COMPARISON_PROMPT_TEMPLATE.format(context=context, question=question)
    response = generate(prompt)

    return {
        "question": question,
        "answer": response,
        "documents_compared": list(per_doc_hits.keys()),
        "hits_by_document": per_doc_hits,
    }


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Compare the deductibles across these policies."
    result = compare(q)
    print(result["answer"])
    print(f"\n(Compared documents: {result['documents_compared']})")
