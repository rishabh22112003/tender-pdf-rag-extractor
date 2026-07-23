"""
STEP 1: PDF Upload (Document Loader) + Text Splitter
"""
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(text):
    """Bilingual PDF (Hindi+English) ka Hindi/garbled part hata dete hain — embedding model English-only hai"""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_split(pdf_path=None):
    """pdf_path=None -> config.py wali fixed PDF. pdf_path=<path> -> dynamic (Streamlit ke liye)"""
    path_to_use = pdf_path or PDF_PATH
    print(f"Loading PDF from: {path_to_use}")

    loader = PyPDFLoader(path_to_use)
    pages = loader.load()
    print(f"Total pages loaded: {len(pages)}")

    for page in pages:
        page.page_content = clean_text(page.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    chunks = load_and_split()
    print("\n--- First 3 chunks (sample) ---\n")
    for i, chunk in enumerate(chunks[:3]):
        print(f"[Chunk {i}] (source page: {chunk.metadata.get('page')})")
        print(chunk.page_content[:200], "...")
        print("-" * 60)
    print(f"\nStep 1 complete. {len(chunks)} chunks ready for embedding (Step 2).")