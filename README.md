# 📄 Tender PDF → Structured Data Extractor

A production-style **Retrieval-Augmented Generation (RAG)** system that reads long, unstructured government tender/EOI PDFs (100–300+ pages) and automatically extracts structured business fields — EMD amount, bid dates, eligibility criteria, financial terms, and more — into an interactive, editable UI with Excel export.

Built to solve a real problem: manually reading lengthy tender documents to pull out 30+ business-critical fields is slow and error-prone. This tool automates it end-to-end while staying cost-efficient and transparent about API usage.

🔗 **Live demo:** https://tender-pdf-extractor.streamlit.app/

---

## 💡 Background & Motivation

This project was built during a one-month AI/ML internship at **Volks Energie Pvt. Ltd.** (June–July 2026). While working there, I observed that employees were manually reading through lengthy (60+ pages) government tender and EOI PDFs to enter structured data (EMD amounts, bid dates, eligibility criteria, etc.) into internal systems — a slow, repetitive, and error-prone process.

This system automates that workflow using an AI-powered document processing pipeline (an approach known as **Intelligent Document Processing / IDP**), which differs from traditional rule-based automation in that it can handle unstructured, variably-formatted documents by understanding meaning rather than relying on fixed templates.

---

## ✨ Features

- **📤 Upload any tender/EOI PDF** — no manual configuration needed
- **🔍 Automatic hyperlink discovery** — detects and downloads PDFs linked *inside* the main document (annexures, category lists, certificates) and merges their content into the same search index
- **🧠 Hybrid retrieval pipeline** — combines semantic search (dense embeddings) with keyword search (BM25) using Reciprocal Rank Fusion, followed by cross-encoder re-ranking for accuracy
- **📋 30+ structured fields extracted automatically** — organized into logical groups (identifiers, financial terms, eligibility, timelines, contact info)
- **✏️ Fully editable form** — any field the model couldn't find is clearly marked "Info not found" and can be filled in manually
- **📎 Linked document viewer** — download buttons + auto-generated 1-2 line summaries for every linked PDF found
- **🔍 Free-form Q&A** — ask any question not covered by the pre-defined fields, answered from the same indexed document
- **📊 Live API usage dashboard** — tracks API calls, tokens consumed, and your **real-time remaining Groq quota** (read directly from API response headers)
- **📥 One-click Excel export** — download all extracted + manually-edited fields as a `.xlsx` file
- **👁️ Built-in PDF preview** — view the original document side-by-side with the extracted data

---

## 🏗️ How It Works (Architecture)

```
                          ┌─────────────────────┐
                          │   PDF Upload (UI)    │
                          └──────────┬───────────┘
                                     ▼
                    ┌────────────────────────────────┐
                    │ Document Loader + Text Cleaning │  (strips noisy/bilingual chars)
                    └────────────────┬───────────────┘
                                     ▼
                    ┌────────────────────────────────┐
                    │  Recursive Text Splitter        │  (chunk_size=800, overlap=100)
                    └────────────────┬───────────────┘
                                     ▼
              ┌──────────────────────────────────────────┐
              │  Hyperlink Extraction (PyMuPDF)           │
              │  → verifies + downloads linked PDFs       │
              │  → merges their chunks into the same set  │
              └──────────────────────┬─────────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────┐
        │  Embeddings — sentence-transformers (local, free) │
        └───────────────┬────────────────────┬─────────────┘
                         ▼                    ▼
              ┌───────────────────┐   ┌───────────────────┐
              │  FAISS Vector DB  │   │   BM25 Keyword DB │
              └─────────┬─────────┘   └─────────┬─────────┘
                         └──────────┬───────────┘
                                    ▼
                  Reciprocal Rank Fusion (RRF)
                                    ▼
                  Cross-Encoder Re-ranking
                                    ▼
                  Neighbor Chunk Expansion
                                    ▼
                  System Prompt Assembly
                                    ▼
                  LLM Call (Groq — Llama 3.3 70B)
                                    ▼
                  Structured JSON field output
                                    ▼
                  Streamlit UI (editable form + preview + Excel export)
```

---

## 🧰 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| PDF parsing | `pypdf` (via LangChain's `PyPDFLoader`) | Reliable, free text extraction |
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) | Splits on natural boundaries with overlap |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Runs **locally, free, no API cost** |
| Vector search | `faiss-cpu` (`IndexFlatL2`) | Fast, free, in-process similarity search |
| Keyword search | `rank_bm25` | Catches exact codes/IDs/numbers embeddings miss |
| Re-ranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Scores query+passage jointly for higher precision |
| LLM | Groq API (Llama 3.3 70B) | Free tier, no credit card, very fast inference |
| Hyperlink extraction | `PyMuPDF` (`fitz`) | Reliable PDF link/annotation parsing |
| UI | `streamlit` | Fast to build, Python-native |
| Excel export | `openpyxl` | Offline, no external service |

---

## 💰 Cost Efficiency

Only the final LLM generation step (Groq API call) costs tokens. Everything else — chunking, embeddings, vector search, keyword search, and re-ranking — runs **locally, for free**.

Additional cost-saving decisions:
- **Batched extraction** — 30+ fields are extracted in ~6-7 grouped LLM calls instead of one call per field
- **Local embeddings** — no per-token embedding API charges
- **Session-level index caching** — the built FAISS/BM25 index is reused for all follow-up questions in a session, avoiding re-processing the PDF
- **Transparent usage tracking** — the UI shows live API call count, token usage, and remaining Groq quota (pulled from real API response headers) so cost is never a surprise

---

## 📂 Project Structure

```
rag_pdf_project/
├── data/                          # PDFs, FAISS index, chunk cache (gitignored)
├── src/
│   ├── config.py                  # Central configuration (paths, models, chunk settings)
│   ├── step1_load_and_split.py    # PDF loading + text cleaning + chunking
│   ├── step2_embeddings.py        # Sentence-transformer embeddings
│   ├── step3_vector_db.py         # FAISS index building
│   ├── step4_semantic_search.py   # Hybrid search + RRF + cross-encoder reranking
│   ├── step5_llm_generate.py      # Prompt assembly + Groq LLM calls + usage tracking
│   ├── extract_to_excel.py        # Structured field extraction → Excel (CLI mode)
│   └── fetch_linked_pdfs.py       # Hyperlink discovery + download
├── main.py                        # CLI: interactive Q&A over a configured PDF
├── streamlit_app.py               # Main UI: upload, extract, preview, Q&A, export
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/rishabh22112003/<your-repo-name>.git
cd <your-repo-name>
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Add your Groq API key

Get a free key at [console.groq.com](https://console.groq.com) (no credit card required), then create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### 3. Run the Streamlit app (recommended)

```bash
streamlit run streamlit_app.py
```

Upload any tender/EOI PDF and the app will handle everything automatically.

### 4. (Optional) CLI mode

Set your PDF filename in `src/config.py` (`PDF_FILENAME`), place the PDF in `data/`, then:

```bash
python main.py
```

For a one-shot Excel export instead of interactive Q&A:

```bash
python src/extract_to_excel.py
```

---

## 🧪 Key Technical Challenges Solved

This project went through several rounds of real debugging — not just a first-pass build:

1. **Keyword-only search failed on paraphrased queries** → switched from TF-IDF to real semantic embeddings.
2. **Semantic search missed exact codes/reference numbers** → added hybrid search (BM25 + semantic via Reciprocal Rank Fusion).
3. **Tables got split across chunk boundaries**, separating headers from data → added neighbor chunk expansion.
4. **Verbose text was outranking short-but-correct answers** (BM25 term-frequency bias) → added cross-encoder re-ranking, which scores query and passage jointly instead of independently.
5. **Bilingual (Hindi + English) PDFs added embedding noise** → added text cleaning to strip non-ASCII content before embedding.
6. **Hyperlinked annexure PDFs inside tenders** were being ignored → added automatic link extraction, content-type verification, and cross-document merging.

---

## ⚠️ Known Limitations

- Uses `IndexFlatL2` (exact search) — fine at this scale, but would need an approximate index (e.g. IVF) or a managed vector DB for millions of vectors.
- Extraction accuracy is currently spot-checked manually; no formal evaluation metric (e.g. RAGAS) yet.
- Groq's free tier has rate limits — a public demo with many concurrent users may occasionally hit them.

---

## 📝 License

This project was built as a personal learning/portfolio project. Feel free to fork and adapt for your own use.

---

## 🙋 Author

**Rishabh Maurya** — Final-year B.Tech CSE (AI specialization), Gurugram University
GitHub: [@rishabh22112003](https://github.com/rishabh22112003)
