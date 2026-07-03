"""
Streamlit UI. Run locally with:  python -m streamlit run app.py
Deploy for free at: https://share.streamlit.io (connect your GitHub repo)

Deployment note: set LLM_BACKEND="groq" and GROQ_API_KEY as Streamlit
Cloud secrets (Settings -> Secrets in the app dashboard). Ollama can't
run on a hosted server since it needs a local background process --
the block below reads those secrets into the environment before the
rest of the pipeline is imported, so it picks up "groq" automatically
when deployed and falls back to "ollama" for local runs where no
secrets file exists.
"""

import os

import streamlit as st

# Must run before importing src.* -- generate() in src/llm.py reads
# LLM_BACKEND from the environment on every call, so this just needs to
# land in os.environ before the first real request, not before import,
# but doing it here keeps all config in one place at the top of the file.
try:
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
except Exception:
    pass  # no secrets.toml locally -- expected, defaults to ollama

import tempfile

from src.index import index_pdf, list_indexed_documents
from src.query import answer_from_doc
from src.compare import compare
from src.exhibit import build_exhibit, render_markdown

st.set_page_config(page_title="Insurance Coverage Copilot", page_icon="📋", layout="wide")
st.title("📋 Insurance Coverage Copilot")
st.caption(
    "Upload one or more commercial insurance policies. Ask questions, compare "
    "across policies, or generate a coverage comparison exhibit."
)

with st.sidebar:
    st.subheader("Upload a policy")
    uploaded = st.file_uploader("PDF", type="pdf", key="uploader")
    carrier = st.text_input("Carrier name", value="", placeholder="e.g. chubb")
    coverage = st.text_input("Coverage type", value="", placeholder="e.g. general_liability")

    if uploaded is not None and st.button("Index this document"):
        doc_id = uploaded.name.replace(".pdf", "")
        # Save under a stable, meaningful filename (not a random tmp name)
        # so doc_id stays consistent across re-runs and matches what the
        # CLI shows in `python main.py list`.
        save_dir = tempfile.gettempdir()
        save_path = os.path.join(save_dir, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner(f"Reading {uploaded.name} and building its index..."):
            n_chunks = index_pdf(
                save_path,
                carrier=carrier or "unknown",
                coverage_type=coverage or "unknown",
            )
        st.success(f"Indexed {n_chunks} sections from '{doc_id}'")

# Refresh the list of indexed documents every render so the UI reflects
# what's actually in Chroma, not just what was uploaded this session.
try:
    indexed_docs = list_indexed_documents()
except Exception:
    indexed_docs = []

if not indexed_docs:
    st.info("Upload and index at least one PDF using the sidebar to get started.")
    st.stop()

with st.expander(f"📁 {len(indexed_docs)} document(s) indexed", expanded=True):
    for d in indexed_docs:
        st.write(f"**{d['doc_id']}** — carrier: {d['carrier']}, coverage: {d['coverage_type']}")

doc_ids = [d["doc_id"] for d in indexed_docs]

tab_ask, tab_compare, tab_exhibit = st.tabs(["Ask", "Compare", "Exhibit"])

with tab_ask:
    st.caption("Ask a question about ONE specific document.")
    selected_doc = st.selectbox("Document", doc_ids, key="ask_doc")
    question = st.text_input("Question", key="ask_question")
    if question:
        with st.spinner("Reading through the document to answer that..."):
            try:
                result = answer_from_doc(question, selected_doc)
                st.markdown(f"**Answer:** {result['answer']}")
                st.caption(f"Retrieved from pages: {result['retrieved_pages']}")
                with st.expander("Show retrieved chunks (debug view)"):
                    for hit in result["chunks_used"]:
                        st.markdown(f"**Page {hit['page']}** (distance: {hit['distance']:.3f})")
                        st.text(hit["text"][:300])
            except Exception as e:
                st.error(f"Generation failed: {e}")

with tab_compare:
    st.caption("Ask a question that compares ACROSS all indexed documents.")
    compare_question = st.text_input("Question", key="compare_question")
    if compare_question:
        with st.spinner("Checking each document and comparing..."):
            try:
                result = compare(compare_question)
                st.markdown(f"**Answer:** {result['answer']}")
                st.caption(f"Compared documents: {result['documents_compared']}")
            except Exception as e:
                st.error(f"Generation failed: {e}")

with tab_exhibit:
    st.caption("Generate a coverage comparison exhibit between two documents.")
    if len(doc_ids) < 2:
        st.info("Index at least 2 documents to generate an exhibit.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            doc_a = st.selectbox("Document A (expiring)", doc_ids, key="exhibit_a")
        with col2:
            doc_b = st.selectbox("Document B (proposed)", [d for d in doc_ids if d != doc_a], key="exhibit_b")

        if st.button("Generate exhibit"):
            with st.spinner("Building the comparison exhibit — checking each coverage aspect, may take a minute..."):
                try:
                    exhibit = build_exhibit(
                        doc_a, doc_b,
                        label_a=f"{doc_a} (expiring)",
                        label_b=f"{doc_b} (proposed)",
                    )
                    markdown = render_markdown(exhibit)
                    st.markdown(markdown)
                    st.download_button(
                        "Download exhibit as .md",
                        data=markdown,
                        file_name=f"exhibit_{doc_a}_vs_{doc_b}.md",
                    )
                except Exception as e:
                    st.error(f"Exhibit generation failed: {e}")
