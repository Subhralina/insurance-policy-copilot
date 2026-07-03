"""
Step 7: Coverage comparison exhibit.

This is the artifact an underwriter or broker actually uses at quote
time: a structured "expiring vs. proposed" table, with gaps explicitly
flagged. Built on top of retrieve_from_doc() rather than compare()'s
general-purpose retrieval, because an exhibit needs the SAME set of
aspects checked against BOTH documents in a fixed order -- a free-form
comparison question doesn't guarantee that symmetry.

IMPORTANT: this drafts the exhibit. It does not replace an
underwriter's review before anything goes to a client -- see the
"human review" note in the generated output.
"""

from src.retrieve import retrieve_from_doc
from src.index import DEFAULT_COLLECTION_NAME
from src.llm import generate

# Default aspects checked in every exhibit. Extend this list for a
# specific coverage type (e.g. add "additional insured", "cyber
# endorsement", "business interruption period" for property/GL).
DEFAULT_ASPECTS = [
    "per-occurrence and aggregate limits",
    "deductible amount",
    "key exclusions",
    "notable endorsements not part of the base form",
]

ROW_PROMPT_TEMPLATE = """Compare how two insurance policy documents address the following aspect: {aspect}

Document A ({label_a}) context:
{context_a}

Document B ({label_b}) context:
{context_b}

Respond in EXACTLY this format, nothing else:
Document A: <what it says, one short sentence, or "not found in retrieved context">
Document B: <what it says, one short sentence, or "not found in retrieved context">
Gap: <"None" if equivalent, or a short description of what's present in one but not the other>
"""


def build_exhibit(
    doc_a_id: str,
    doc_b_id: str,
    label_a: str = None,
    label_b: str = None,
    aspects: list[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> dict:
    label_a = label_a or doc_a_id
    label_b = label_b or doc_b_id
    aspects = aspects or DEFAULT_ASPECTS

    rows = []
    for aspect in aspects:
        hits_a = retrieve_from_doc(aspect, doc_a_id, k=2, collection_name=collection_name)
        hits_b = retrieve_from_doc(aspect, doc_b_id, k=2, collection_name=collection_name)

        context_a = "\n".join(f"[p.{h['page']}] {h['text']}" for h in hits_a) or "(no relevant chunks retrieved)"
        context_b = "\n".join(f"[p.{h['page']}] {h['text']}" for h in hits_b) or "(no relevant chunks retrieved)"

        prompt = ROW_PROMPT_TEMPLATE.format(
            aspect=aspect, label_a=label_a, label_b=label_b, context_a=context_a, context_b=context_b
        )
        response = generate(prompt)
        rows.append({"aspect": aspect, "raw": response})

    return {"label_a": label_a, "label_b": label_b, "rows": rows}


def render_markdown(exhibit: dict) -> str:
    lines = [
        f"# Coverage Comparison Exhibit",
        f"**{exhibit['label_a']}** vs. **{exhibit['label_b']}**\n",
        "> Drafted automatically from indexed policy documents. Requires underwriter review before use with a client.\n",
        "| Aspect | " + exhibit["label_a"] + " | " + exhibit["label_b"] + " | Gap |",
        "|---|---|---|---|",
    ]
    for row in exhibit["rows"]:
        # Parse the fixed-format LLM response back into table cells
        a, b, gap = "—", "—", "—"
        for line in row["raw"].splitlines():
            if line.startswith("Document A:"):
                a = line.replace("Document A:", "").strip()
            elif line.startswith("Document B:"):
                b = line.replace("Document B:", "").strip()
            elif line.startswith("Gap:"):
                gap = line.replace("Gap:", "").strip()
        flag = "**GAP**" if gap.lower() != "none" else "None"
        lines.append(f"| {row['aspect']} | {a} | {b} | {flag}: {gap} |")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    doc_a = sys.argv[1] if len(sys.argv) > 1 else "gl_hartford"
    doc_b = sys.argv[2] if len(sys.argv) > 2 else "gl_hiscox"
    exhibit = build_exhibit(doc_a, doc_b, label_a=f"{doc_a} (expiring)", label_b=f"{doc_b} (proposed)")
    print(render_markdown(exhibit))
