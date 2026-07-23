"""
STEP 4: Hybrid Search (BM25 + Semantic) + Cross-Encoder Re-ranking + Neighbor Expansion

WHY HYBRID: Dense embeddings capture MEANING well, but are weak at exact tokens like
IDs/codes/numbers. BM25 (keyword search) catches exact term matches that embeddings miss.
Reciprocal Rank Fusion (RRF) combines both rankings into one.

WHY RERANKING: BM25/embeddings can rank a verbose-but-wrong chunk above a short-but-correct
one (term-frequency bias). Cross-encoder scores (query, chunk) together for real relevance.

WHY NEIGHBOR EXPANSION: Tables/headings often get split across adjacent chunks — including
neighbors ensures the full table/section reaches the LLM.
"""
import faiss
import pickle
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config import FAISS_INDEX_PATH, CHUNKS_CACHE_PATH, TOP_K, FETCH_K, RERANKER_MODEL
from step2_embeddings import get_model

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"Loading re-ranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def load_index_and_chunks():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_CACHE_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9/\-\.]+", text.lower())

def build_bm25_index(chunks):
    tokenized_corpus = [tokenize(c.page_content) for c in chunks]
    return BM25Okapi(tokenized_corpus)

def semantic_search_raw(query, index, model, top_k):
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)
    return list(zip(indices[0].tolist(), distances[0].tolist()))

def bm25_search_raw(query, bm25, top_k):
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]

def reciprocal_rank_fusion(semantic_results, bm25_results, k=60):
    rrf_scores = {}
    for rank, (idx, _) in enumerate(semantic_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
    for rank, (idx, _) in enumerate(bm25_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

def rerank_with_cross_encoder(query, candidate_indices, chunks, top_k):
    reranker = get_reranker()
    pairs = [[query, chunks[idx].page_content] for idx in candidate_indices]
    scores = reranker.predict(pairs)
    scored = list(zip(candidate_indices, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def expand_with_neighbors(top_chunk_indices, total_chunks, window=1):
    expanded = set()
    for idx in top_chunk_indices:
        for offset in range(-window, window + 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < total_chunks:
                expanded.add(neighbor_idx)
    return sorted(expanded)

def hybrid_search(query, index, chunks, bm25, top_k=TOP_K, fetch_k=FETCH_K, expand_neighbors=True):
    model = get_model()

    semantic_results = semantic_search_raw(query, index, model, fetch_k)
    bm25_results = bm25_search_raw(query, bm25, fetch_k)
    fused = reciprocal_rank_fusion(semantic_results, bm25_results)
    candidate_indices = [idx for idx, _ in fused]

    reranked = rerank_with_cross_encoder(query, candidate_indices, chunks, top_k)
    top_indices = [idx for idx, _ in reranked]
    score_map = dict(reranked)

    if expand_neighbors:
        final_indices = expand_with_neighbors(top_indices, len(chunks), window=1)
    else:
        final_indices = top_indices

    results = []
    for rank, idx in enumerate(final_indices, start=1):
        results.append({
            "rank": rank,
            "chunk_idx": idx,
            "page": chunks[idx].metadata.get("page"),
            "rerank_score": round(float(score_map.get(idx, 0)), 4),
            "content": chunks[idx].page_content
        })
    return results

def semantic_search(query, index, chunks, top_k=TOP_K):
    bm25 = build_bm25_index(chunks)
    return hybrid_search(query, index, chunks, bm25, top_k=top_k)

if __name__ == "__main__":
    index, chunks = load_index_and_chunks()
    bm25 = build_bm25_index(chunks)
    user_query = input("Test query daalo: ")
    results = hybrid_search(user_query, index, chunks, bm25)
    for r in results:
        print(f"[chunk_idx={r['chunk_idx']}] Page {r['page']} | score={r['rerank_score']}")
        print(f"   {r['content'][:150]}...")