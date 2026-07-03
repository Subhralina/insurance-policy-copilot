"""
Step 5: Answering.

Combines retrieval + generation. The prompt is written to do two
important things a naive "stuff it all in and ask" approach usually
skips:
  1. Force the model to say it doesn't know if the retrieved chunks
     don't actually answer the question, instead of hallucinating.
  2. Force it to cite which page(s) it used, so an answer is
     traceable back to source -- this is the difference between a
     retrieval system and a chatbot that sounds confident.
"""

from src.retrieve import retrieve, retrieve_from_doc
from src.llm import generate

PROMPT_TEMPLATE = """You are answering a question using ONLY the context below, \
extracted from a PDF. Each context piece is labeled with its page number.

If the context does not contain enough information to answer, say exactly:
"I can't find that in the document." Do not use outside knowledge.

When you do answer, end with a line like: Source: page {{page numbers}}

Context:
{context}

Question: {question}

Answer:"""


def answer(question: str, k: int = 4) -> dict:
    hits = retrieve(question, k=k)
    context = "\n\n".join(f"[page {h['page']}] {h['text']}" for h in hits)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = generate(prompt)
    return {
        "question": question,
        "answer": response,
        "retrieved_pages": sorted({h["page"] for h in hits}),
        "chunks_used": hits,
    }


def answer_from_doc(question: str, doc_id: str, k: int = 4) -> dict:
    """Same as answer(), but scoped to a single indexed document -- needed
    once more than one document is indexed, otherwise retrieval searches
    across all of them and can return chunks from the wrong policy."""
    hits = retrieve_from_doc(question, doc_id, k=k)
    context = "\n\n".join(f"[page {h['page']}] {h['text']}" for h in hits)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = generate(prompt)
    return {
        "question": question,
        "answer": response,
        "retrieved_pages": sorted({h["page"] for h in hits}),
        "chunks_used": hits,
    }


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    result = answer(q)
    print(result["answer"])
    print(f"\n(Retrieved from pages: {result['retrieved_pages']})")
