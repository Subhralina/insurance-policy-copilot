"""
Step 3: Retrieval.

Given a question, embed it with the *same* model used for indexing,
then ask Chroma for the nearest chunks. This is where you can measure
retrieval quality independent of the LLM -- if the right chunk isn't
in the top-k results, no amount of clever prompting downstream fixes it.
"""

from src.index import get_collection, get_embedder


def retrieve(query: str, k: int = 4) -> list[dict]:
    embedder = get_embedder()
    collection = get_collection(reset=False)

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "page": results["metadatas"][0][i]["page"],
                "distance": results["distances"][0][i],
            }
        )
    return hits


def retrieve_from_doc(query: str, doc_id: str, k: int = 3) -> list[dict]:
    """Same as retrieve(), but scoped to a single indexed document by doc_id."""
    embedder = get_embedder()
    collection = get_collection(reset=False)

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where={"doc_id": doc_id},
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "text": results["documents"][0][i],
                "page": results["metadatas"][0][i]["page"],
                "distance": results["distances"][0][i],
            }
        )
    return hits


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    for hit in retrieve(q):
        print(f"[page {hit['page']}, dist {hit['distance']:.3f}] {hit['text'][:150]}...")
