"""
CLI entry point.

Usage:
    python main.py index data/your_file.pdf
    python main.py ask "your question here"
"""

import sys

from src.index import index_pdf, list_indexed_documents
from src.query import answer
from src.compare import compare
from src.exhibit import build_exhibit, render_markdown


def main():
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python main.py index <pdf_path> [carrier] [coverage_type]\n"
            "  python main.py ask <question>              (single most recent doc)\n"
            "  python main.py compare <question>          (across all indexed docs)\n"
            "  python main.py exhibit <doc_a_id> <doc_b_id> [output_path]  (coverage comparison exhibit)\n"
            "  python main.py list                        (show indexed docs)"
        )
        return

    command = sys.argv[1]

    if command == "index":
        pdf_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample.pdf"
        carrier = sys.argv[3] if len(sys.argv) > 3 else "unknown"
        coverage = sys.argv[4] if len(sys.argv) > 4 else "unknown"
        n = index_pdf(pdf_path, carrier=carrier, coverage_type=coverage)
        print(f"Indexed {n} chunks from {pdf_path} (carrier={carrier}, coverage={coverage})")

    elif command == "list":
        docs = list_indexed_documents()
        if not docs:
            print("No documents indexed yet.")
        for d in docs:
            print(f"  {d['doc_id']}  |  carrier: {d['carrier']}  |  coverage: {d['coverage_type']}")

    elif command == "ask":
        question = " ".join(sys.argv[2:]) or "What is this document about?"
        result = answer(question)
        print(f"\nQ: {question}\n")
        print(f"A: {result['answer']}")
        print(f"\n(Retrieved from pages: {result['retrieved_pages']})")

    elif command == "compare":
        question = " ".join(sys.argv[2:]) or "Compare these policies."
        result = compare(question)
        print(f"\nQ: {question}\n")
        print(f"A: {result['answer']}")
        print(f"\n(Compared documents: {result['documents_compared']})")

    elif command == "exhibit":
        if len(sys.argv) < 4:
            print("Usage: python main.py exhibit <doc_a_id> <doc_b_id> [output_path]")
            return
        doc_a, doc_b = sys.argv[2], sys.argv[3]
        output_path = sys.argv[4] if len(sys.argv) > 4 else None

        exhibit = build_exhibit(doc_a, doc_b, label_a=f"{doc_a} (expiring)", label_b=f"{doc_b} (proposed)")
        markdown = render_markdown(exhibit)
        print(markdown)

        if output_path:
            with open(output_path, "w") as f:
                f.write(markdown)
            print(f"\nSaved to {output_path}")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
