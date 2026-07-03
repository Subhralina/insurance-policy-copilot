"""
Step 2: Indexing.

Embeds each chunk with a local sentence-transformers model (free, no
API key, runs on CPU) and stores the vectors in a local Chroma
collection on disk. This is the retrieval half of RAG -- get this
wrong and the generation step never has a chance.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingest import build_chunks

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free, good enough to start
DB_PATH = "chroma_db"
COLLECTION_NAME = "pdf_chunks"


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=DB_PATH)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION_NAME)


def index_pdf(
    pdf_path: str,
    carrier: str = "unknown",
    coverage_type: str = "unknown",
    reset: bool = False,
) -> int:
    """
    Chunk a PDF, embed every chunk, and write it into the vector store.

    reset defaults to False now -- each call ADDS a document to the
    collection instead of replacing it, which is what multi-document
    comparison needs. Every chunk is tagged with carrier and
    coverage_type so retrieval can later filter or group by document.

    Chunk IDs are prefixed with the pdf filename so re-indexing the
    same file overwrites its own chunks instead of colliding with a
    different document's chunk_0, chunk_1, etc.
    """
    chunks = build_chunks(pdf_path)
    if not chunks:
        raise ValueError(f"No extractable text found in {pdf_path}")

    embedder = get_embedder()
    collection = get_collection(reset=reset)

    doc_id = pdf_path.split("/")[-1].replace(".pdf", "")
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids=[f"{doc_id}__{c.id}" for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "page": c.page,
                "source": pdf_path,
                "doc_id": doc_id,
                "carrier": carrier,
                "coverage_type": coverage_type,
            }
            for c in chunks
        ],
    )
    return len(chunks)


def list_indexed_documents() -> list[dict]:
    """Return the distinct documents currently in the collection, with metadata."""
    collection = get_collection(reset=False)
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    seen = {}
    for m in all_meta:
        seen[m["doc_id"]] = {"doc_id": m["doc_id"], "carrier": m["carrier"], "coverage_type": m["coverage_type"]}
    return list(seen.values())


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    carrier = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    coverage = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    n = index_pdf(path, carrier=carrier, coverage_type=coverage)
    print(f"Indexed {n} chunks from {path} (carrier={carrier}, coverage={coverage})")
