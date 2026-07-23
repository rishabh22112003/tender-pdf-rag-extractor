"""
STEP 3: Vector Database (FAISS)
"""
import faiss
import numpy as np
import pickle
from step1_load_and_split import load_and_split
from step2_embeddings import generate_embeddings
from config import FAISS_INDEX_PATH, CHUNKS_CACHE_PATH

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)   # exact nearest-neighbor search
    index.add(embeddings)
    return index

def build_and_save():
    chunks = load_and_split()
    embeddings, model = generate_embeddings(chunks)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_CACHE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks

if __name__ == "__main__":
    index, chunks = build_and_save()
    print(f"\nFAISS index built successfully")
    print(f"Total vectors stored: {index.ntotal}")
    print(f"Saved index to: {FAISS_INDEX_PATH}")
    print(f"Saved chunks cache to: {CHUNKS_CACHE_PATH}")
    print(f"Step 3 complete.")