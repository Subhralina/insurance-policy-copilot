# insurance-coverage-copilot

A retrieval-augmented assistant for commercial insurance policy documents —
upload policies, ask questions, get answers grounded in the document with
page citations, and generate coverage comparison exhibits across carriers.

Built to actually understand each piece of a RAG pipeline, not to wrap
an API call.

## How it works

1. **Ingest** (`src/ingest.py`) — extract text from the PDF, split into
   overlapping chunks (800 chars, 150 overlap).
2. **Index** (`src/index.py`) — embed each chunk locally with
   `sentence-transformers` (all-MiniLM-L6-v2), store in a local Chroma
   vector database.
3. **Retrieve** (`src/retrieve.py`) — embed the question, pull the
   nearest chunks from Chroma.
4. **Generate** (`src/query.py` + `src/llm.py`) — pass retrieved chunks
   to an LLM with a prompt that forces citation and forbids answering
   outside the retrieved context.

## Data source

Real specimen commercial insurance policy forms, publicly posted by
carriers and government agencies for reference:

- GL (Hiscox): https://www.hiscox.com/documents/partner-agent/specimen_forms/GL_Form.pdf
- GL (Hartford): https://www.insurancebee.com/documents/wordings/general-liability-policy-form.pdf
- Property (ISO CP 00 10, via NY OGS): https://ogs.ny.gov/system/files/documents/2021/09/cp10300917-sample.pdf
- Workers' Comp (Hartford/Insurance Board): https://www.insuranceboard.org/wp-content/uploads/dlm_uploads/2019/11/10-sample-workers-compensation-policy.pdf

These are ISO-copyrighted base forms published for public reference —
fine for a personal/portfolio project, not for redistribution.

## Setup (local, 100% free)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Ollama (https://ollama.com) and pull a small model:
ollama pull llama3.2

# Index a PDF and ask a question:
python main.py index data/your_file.pdf
python main.py ask "What is this document about?"

# Or run the UI:
streamlit run app.py
```

## Multi-document comparison

```bash
# Index several policies, tagging each with carrier and coverage type:
python main.py index data/gl_hiscox.pdf hiscox general_liability
python main.py index data/gl_hartford.pdf hartford general_liability

# Confirm what's indexed:
python main.py list

# Ask a question that compares across all indexed documents:
python main.py compare "Compare the deductibles across these policies"
```

`compare` retrieves the top matches from EACH indexed document
separately before generating an answer, rather than one flattened
top-k search — otherwise the single best-matching document crowds
out the others and the comparison silently ignores documents that
were a slightly weaker semantic match. This is the real failure mode
naive RAG hits the moment you index more than one document, and the
fix is in `src/compare.py`.

## Deploying for free

1. Push this repo to GitHub.
2. Get a free API key at [console.groq.com](https://console.groq.com).
3. Deploy on [Streamlit Community Cloud](https://share.streamlit.io),
   connect the repo, and add `GROQ_API_KEY` under app secrets.
4. In the app sidebar, switch the backend to `groq` (Ollama can't run
   on a hosted server — it needs a local process).

## Coverage comparison exhibit (the quoting-workflow feature)

This is the artifact an underwriter actually produces when quoting a
new client switching carriers: an "expiring vs. proposed" table with
gaps flagged, checked against a fixed set of aspects so both documents
are compared symmetrically -- not a free-form Q&A answer.

```bash
python main.py exhibit gl_hartford gl_hiscox exhibit_output.md
```

`gl_hartford` / `gl_hiscox` are the doc_id values shown by `python
main.py list` (the PDF filename without `.pdf`). This checks limits,
deductible, key exclusions, and notable endorsements by default --
edit `DEFAULT_ASPECTS` in `src/exhibit.py` to check different things
for a different coverage type (e.g. add "business interruption
period" for property).

**This drafts the exhibit. It does not replace underwriter review** --
that's stated directly in the generated output, and it's a real
scope boundary worth keeping, not just a disclaimer for show: a
mis-flagged coverage gap sent to a client uncaught is a real liability,
and a tool that quietly implied it didn't need review would be the
wrong thing to build here.

## What I'd measure next (eval)

- [ ] Build a 20–30 question test set with known correct answers.
- [ ] Measure retrieval precision: was the right chunk in the top-k?
- [ ] Measure answer correctness separately from retrieval — did the
      LLM use the right chunk correctly once it had it?
- [ ] Try a reranking step and see if precision actually improves.

## Project narrative (fill this in as you build, not after)

**Problem:** _[one concrete sentence — what can't you do without this?]_

**Naive baseline:** _[what happened when you just stuffed the whole
doc into the prompt? Where did it break?]_

**Design decisions:** _[why 800/150 for chunk size? why MiniLM over a
bigger embedding model? what would you change with more time?]_

**What broke:** _[a real retrieval miss or hallucination you hit —
the specific question and what went wrong]_

**Numbers:** _[retrieval precision, or even just "N/M eval questions
answered correctly with correct citations"]_
