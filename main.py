"""
MAIN: Smart auto-build + interactive Q&A
Pehli baar chalane pe khud PDF process karega, agli baar seedha Q&A shuru karega.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import FAISS_INDEX_PATH, CHUNKS_CACHE_PATH, TOP_K


def ensure_index_built():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_CACHE_PATH):
        print("Existing index mil gaya.\n")
        return
    print("Index nahi mila — PDF process kar raha hoon...\n")
    from step3_vector_db import build_and_save
    build_and_save()
    print("\nIndex ban gaya.\n")


def main():
    ensure_index_built()
    from step5_llm_generate import rag_pipeline

    print("RAG Q&A -- apna sawaal poocho ('exit' likho band karne ke liye)\n")
    while True:
        user_query = input("Apna sawaal poocho: ")
        if user_query.strip().lower() in ("exit", "quit"):
            print("Bye!")
            break
        if not user_query.strip():
            continue
        system_query, answer = rag_pipeline(user_query, top_k=TOP_K)
        print(f"\nFINAL OUTPUT:\n{answer}\n")


if __name__ == "__main__":
    main()