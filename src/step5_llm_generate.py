"""
STEP 5: System Query Assembly + Brain (LLM API)
Setup: .env (local) ya st.secrets (Streamlit Cloud) mein GROQ_API_KEY daalo
"""
import os
from dotenv import load_dotenv
from groq import Groq

from step4_semantic_search import load_index_and_chunks, build_bm25_index, hybrid_search
from config import TOP_K, LLM_MODEL

load_dotenv()

# Cumulative usage jo hum khud track karte hain (is session ke liye)
_usage_stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# Groq API se live rate-limit info (har response ke headers se aata hai — ye batata hai
# tumhare ACTUAL account ka kitna quota bacha hai, sirf humara khud ka counter nahi)
_rate_limit_info = {}

def reset_usage_stats():
    """Naya PDF process karne se pehle counter reset karo"""
    global _usage_stats
    _usage_stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def get_usage_stats():
    return dict(_usage_stats)

def get_rate_limit_info():
    """Groq ke response headers se mila live quota info (remaining requests/tokens)"""
    return dict(_rate_limit_info)


def get_api_key():
    """Local: .env se. Streamlit Cloud: st.secrets se (Secrets Manager)."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def build_system_query(user_query, retrieved_chunks):
    """Diagram ka 'System Query' box: Pages + User Query ko combine karna"""
    context = "\n\n".join([f"[Page {c['page']}]: {c['content']}" for c in retrieved_chunks])
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."
If multiple values or clauses appear, pick the one most directly relevant to the question.

Context:
{context}

Question: {user_query}

Answer:"""
    return prompt


def call_llm(prompt):
    """Diagram ka 'Brain (LLM API)' box"""
    api_key = get_api_key()
    if not api_key:
        return "[SKIPPED: GROQ_API_KEY not set. Add it to .env (local) or Secrets (cloud).]"

    client = Groq(api_key=api_key)

    # with_raw_response se hume actual HTTP headers milte hain (rate-limit info ke liye)
    raw_response = client.chat.completions.with_raw_response.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    response = raw_response.parse()

    # Token usage track karo (humara khud ka cumulative counter)
    try:
        usage = response.usage
        _usage_stats["calls"] += 1
        _usage_stats["prompt_tokens"] += usage.prompt_tokens
        _usage_stats["completion_tokens"] += usage.completion_tokens
        _usage_stats["total_tokens"] += usage.total_tokens
    except Exception:
        pass

    # Groq ke actual rate-limit headers capture karo (live remaining quota)
    try:
        headers = raw_response.headers
        _rate_limit_info["limit_requests"] = headers.get("x-ratelimit-limit-requests")
        _rate_limit_info["remaining_requests"] = headers.get("x-ratelimit-remaining-requests")
        _rate_limit_info["limit_tokens"] = headers.get("x-ratelimit-limit-tokens")
        _rate_limit_info["remaining_tokens"] = headers.get("x-ratelimit-remaining-tokens")
        _rate_limit_info["reset_requests"] = headers.get("x-ratelimit-reset-requests")
        _rate_limit_info["reset_tokens"] = headers.get("x-ratelimit-reset-tokens")
    except Exception:
        pass

    return response.choices[0].message.content


def rag_pipeline(user_query, top_k=TOP_K):
    index, chunks = load_index_and_chunks()
    bm25 = build_bm25_index(chunks)
    retrieved = hybrid_search(user_query, index, chunks, bm25, top_k=top_k)
    system_query = build_system_query(user_query, retrieved)
    answer = call_llm(system_query)
    return system_query, answer


if __name__ == "__main__":
    print("RAG Q&A -- 'exit' ya 'quit' likho band karne ke liye\n")
    while True:
        user_query = input("write a query: ")
        if user_query.strip().lower() in ("exit", "quit"):
            print("Exited. Bye!")
            break
        if not user_query.strip():
            continue
        system_query, answer = rag_pipeline(user_query)
        print(f"\nFINAL OUTPUT:\n{answer}\n")

        # Debug: agar rate-limit fields None dikhein, ye pura headers dict dekh lo
        print("\n[DEBUG] Rate limit info:", get_rate_limit_info())