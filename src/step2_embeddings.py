"""
STEP 2: Embeddings (real semantic embeddings using sentence-transformers)
Pehli baar chalane pe model download hoga (~90MB), phir cache ho jaata hai.
"""
from sentence_transformers import SentenceTransformer
from step1_load_and_split import load_and_split
from config import EMBEDDING_MODEL
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def generate_embeddings(chunks):
    model = get_model()
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype(np.float32), model

if __name__ == "__main__":
    chunks = load_and_split()
    embeddings, model = generate_embeddings(chunks)
    print(f"\nEmbedding matrix shape: {embeddings.shape}")
    print(f"Step 2 complete. Embeddings ready for Vector DB (Step 3).")