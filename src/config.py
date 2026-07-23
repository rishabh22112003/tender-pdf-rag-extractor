"""
CONFIG — Sabse pehle isko edit karo

Apni PDF ko 'data' folder mein rakh do, phir neeche PDF_FILENAME mein uska naam likh do.
Baaki saari step files isi config ko import karengi.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FILENAME = "GeM-Bidding-7876499.pdf_1748691368_85813321.pdf"    # <-- CLI scripts (main.py, extract_to_excel.py) ke liye
PDF_PATH = os.path.join(BASE_DIR, "..", "data", PDF_FILENAME)

# Chunking settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval settings
TOP_K = 12
FETCH_K = 30          # kitne candidates initial retrieval se lene hain (reranking se pehle)

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Saved artifacts (CLI use ke liye)
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "data", "faiss_index.bin")
CHUNKS_CACHE_PATH = os.path.join(BASE_DIR, "..", "data", "chunks_cache.pkl")