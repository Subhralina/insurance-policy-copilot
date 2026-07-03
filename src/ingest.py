"""
Step 1: Ingestion.

Takes a PDF, extracts raw text page by page, then splits it into
overlapping chunks. Chunk size and overlap are the first real design
decisions in a RAG system -- too big and retrieval returns noisy
context, too small and you lose the surrounding meaning of a sentence.
"""

from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class Chunk:
    id: str
    text: str
    page: int


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text)."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Simple sliding-window chunker over raw characters.

    chunk_size=800, overlap=150 is a reasonable starting point for
    dense prose (roughly 150-200 words per chunk). Overlap prevents a
    sentence from being cut in half right at a chunk boundary and
    losing its context entirely.
    """
    chunks = []
    start = 0
    text = " ".join(text.split())  # normalize whitespace
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start <= 0:
            break
    return chunks


def build_chunks(pdf_path: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    pages = extract_pages(pdf_path)
    all_chunks: list[Chunk] = []
    chunk_idx = 0
    for page_num, page_text in pages:
        for piece in chunk_text(page_text, chunk_size, overlap):
            all_chunks.append(Chunk(id=f"chunk_{chunk_idx}", text=piece, page=page_num))
            chunk_idx += 1
    return all_chunks


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    chunks = build_chunks(path)
    print(f"Extracted {len(chunks)} chunks from {path}")
    if chunks:
        print("\n--- First chunk preview ---")
        print(f"[page {chunks[0].page}] {chunks[0].text[:200]}...")
