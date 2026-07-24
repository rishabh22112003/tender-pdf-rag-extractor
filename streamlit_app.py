"""
STREAMLIT UI: PDF upload -> auto-extract tender fields -> editable form + PDF preview + Q&A
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import sys, os, json, time
import fitz  # PyMuPDF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from step1_load_and_split import load_and_split
from step2_embeddings import generate_embeddings
from step3_vector_db import build_faiss_index
from step4_semantic_search import build_bm25_index, hybrid_search
from step5_llm_generate import call_llm, build_system_query, reset_usage_stats, get_usage_stats, get_rate_limit_info

from extract_to_excel import FIELD_GROUPS, build_extraction_prompt, clean_json_response
from fetch_linked_pdfs import fetch_all_linked_pdfs

st.set_page_config(page_title="Tender PDF Extractor", layout="wide")

REQUIRED_FIELDS = {"emd_amount", "bid_validity_days", "tender_value", "pbg_percentage", "sd_percentage"}
SAMPLE_PDFS_DIR = os.path.join(os.path.dirname(__file__), "sample_pdfs")


def humanize(key):
    return key.replace("_", " ").title()


def summarize_linked_pdf(path):
    try:
        chunks = load_and_split(pdf_path=path)
        preview_text = " ".join([c.page_content for c in chunks[:3]])[:2000]
        prompt = f"""Summarize what this document is about in exactly 1-2 short sentences (under 30 words).
Document text:
{preview_text}

Summary:"""
        return call_llm(prompt).strip()
    except Exception as e:
        return f"(Could not generate summary: {e})"


def process_pdf(pdf_bytes, filename):
    reset_usage_stats()

    tmp_path = os.path.join("data", f"_uploaded_{filename}")
    os.makedirs("data", exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(pdf_bytes)

    progress = st.progress(0, text="Loading PDF...")
    chunks = load_and_split(pdf_path=tmp_path)

    progress.progress(15, text="Checking for linked PDFs inside document...")
    linked_paths = []
    try:
        linked_dir = os.path.join("data", "linked_pdfs", filename.replace(".pdf", ""))
        linked_paths = fetch_all_linked_pdfs(pdf_path=tmp_path, save_dir=linked_dir)
        for linked_path in linked_paths:
            try:
                linked_chunks = load_and_split(pdf_path=linked_path)
                for c in linked_chunks:
                    c.metadata["source_file"] = os.path.basename(linked_path)
                chunks.extend(linked_chunks)
            except Exception as e:
                st.warning(f"Failed to process linked PDF ({os.path.basename(linked_path)}): {e}")
    except Exception as e:
        st.warning(f"Error while checking linked PDFs (skipping): {e}")

    linked_summaries = {}
    if linked_paths:
        progress.progress(20, text="Summarizing linked documents...")
        for path in linked_paths:
            linked_summaries[path] = summarize_linked_pdf(path)

    progress.progress(25, text="Generating embeddings (model may download on first run)...")
    embeddings, model = generate_embeddings(chunks)

    progress.progress(50, text="Building vector index...")
    index = build_faiss_index(embeddings)
    bm25 = build_bm25_index(chunks)

    progress.progress(65, text="Extracting tender fields...")
    all_data = {}
    total_groups = len(FIELD_GROUPS)
    for i, (group_name, group_config) in enumerate(FIELD_GROUPS.items()):
        retrieved = hybrid_search(group_config["query"], index, chunks, bm25, top_k=10)
        context = "\n\n".join([f"[Page {c['page']}]: {c['content']}" for c in retrieved])
        prompt = build_extraction_prompt(group_config["fields"], context)
        response = call_llm(prompt)
        try:
            data = json.loads(clean_json_response(response))
        except Exception:
            data = {key: None for key in group_config["fields"]}
        all_data.update(data)
        progress.progress(65 + int(30 * (i + 1) / total_groups), text=f"Extracted: {group_name}")

    usage_stats = get_usage_stats()

    progress.progress(100, text="Done!")
    time.sleep(0.3)
    progress.empty()

    return all_data, tmp_path, linked_paths, linked_summaries, usage_stats, index, chunks, bm25


def render_pdf_preview(pdf_path):
    """PyMuPDF se page-by-page image render karta hai — iframe/data-URI browser blocking se bachne ke liye"""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    page_num = st.number_input(
        f"Page (1 to {total_pages})",
        min_value=1, max_value=total_pages, value=1, step=1,
        key=f"preview_page_{os.path.basename(pdf_path)}"
    )

    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=120)
    img_bytes = pix.tobytes("png")

    st.image(img_bytes, use_container_width=True)
    doc.close()


st.title("📄 Tender PDF → Structured Data Extractor")

if os.path.isdir(SAMPLE_PDFS_DIR):
    sample_files = sorted([f for f in os.listdir(SAMPLE_PDFS_DIR) if f.lower().endswith(".pdf")])
else:
    sample_files = []

if sample_files:
    with st.expander("📥 Don't have a tender PDF handy? Download a sample to try", expanded=False):
        st.caption("Download one of these, then upload it below to see the extraction in action.")
        for fname in sample_files:
            fpath = os.path.join(SAMPLE_PDFS_DIR, fname)
            with open(fpath, "rb") as f:
                st.download_button(
                    label=f"⬇️ {fname}",
                    data=f.read(),
                    file_name=fname,
                    mime="application/pdf",
                    key=f"sample_download_{fname}"
                )

uploaded_file = st.file_uploader("PDF (Tender) Upload", type=["pdf"])
st.caption("💡 You can drag and drop a PDF directly onto the box above, or click to browse.")

active_pdf_bytes = None
active_filename = None

if uploaded_file is not None:
    active_pdf_bytes = uploaded_file.getvalue()
    active_filename = uploaded_file.name

if active_pdf_bytes is not None:
    file_key = active_filename

    if st.session_state.get("processed_filename") != file_key:
        with st.spinner("Processing PDF..."):
            data, tmp_path, linked_paths, linked_summaries, usage_stats, index, chunks, bm25 = process_pdf(
                active_pdf_bytes, file_key
            )
        st.session_state["extracted_data"] = data
        st.session_state["pdf_path"] = tmp_path
        st.session_state["linked_paths"] = linked_paths
        st.session_state["linked_summaries"] = linked_summaries
        st.session_state["usage_stats"] = usage_stats
        st.session_state["processed_filename"] = file_key
        st.session_state["index"] = index
        st.session_state["chunks"] = chunks
        st.session_state["bm25"] = bm25
        st.session_state["qa_history"] = []
        st.success("✅ Extraction complete!")

    linked_paths = st.session_state.get("linked_paths", [])
    if linked_paths:
        st.info(f"📎 {len(linked_paths)} linked PDF(s) found and merged with this document.")

    usage = st.session_state.get("usage_stats", {})
    if usage:
        st.subheader("📊 API Usage (for this PDF)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("API Calls", usage.get("calls", 0))
        col2.metric("Prompt Tokens", usage.get("prompt_tokens", 0))
        col3.metric("Completion Tokens", usage.get("completion_tokens", 0))
        col4.metric("Total Tokens", usage.get("total_tokens", 0))

        rate_info = get_rate_limit_info()
        if rate_info.get("remaining_requests") or rate_info.get("remaining_tokens"):
            st.caption("📡 **Live Groq account quota** (from API response headers):")
            rcol1, rcol2 = st.columns(2)
            rcol1.metric("Requests Remaining", f"{rate_info.get('remaining_requests', '?')} / {rate_info.get('limit_requests', '?')}")
            rcol2.metric("Tokens Remaining", f"{rate_info.get('remaining_tokens', '?')} / {rate_info.get('limit_tokens', '?')}")
        else:
            st.caption("💡 Groq free tier (llama-3.3-70b-versatile): ~12,000 tokens/min, 30 requests/min, "
                       "1000 requests/day (approximate — check console.groq.com for your exact limits).")

    st.subheader("🔍 Ask a specific question")
    st.caption("If you need information not covered in the fields below, ask here.")

    qa_col1, qa_col2 = st.columns([4, 1])
    with qa_col1:
        user_question = st.text_input("Type your question:", key="qa_input", label_visibility="collapsed",
                                        placeholder="e.g. What is the delivery location?")
    with qa_col2:
        ask_clicked = st.button("Ask", use_container_width=True)

    if ask_clicked and user_question.strip():
        with st.spinner("Searching document..."):
            index = st.session_state["index"]
            chunks = st.session_state["chunks"]
            bm25 = st.session_state["bm25"]

            retrieved = hybrid_search(user_question, index, chunks, bm25, top_k=10)
            system_query = build_system_query(user_question, retrieved)
            answer = call_llm(system_query)

            st.session_state["qa_history"].insert(0, {"question": user_question, "answer": answer})
            st.session_state["usage_stats"] = get_usage_stats()

    for qa in st.session_state.get("qa_history", []):
        with st.container(border=True):
            st.markdown(f"**Q: {qa['question']}**")
            st.write(qa["answer"])

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Extracted Fields")
        data = st.session_state["extracted_data"]
        form_values = {}

        for group_name, group_config in FIELD_GROUPS.items():
            with st.expander(humanize(group_name), expanded=True):
                for field_key, field_desc in group_config["fields"].items():
                    label = humanize(field_key)
                    if field_key in REQUIRED_FIELDS:
                        label += " *"

                    extracted_value = data.get(field_key)
                    default_value = str(extracted_value) if extracted_value not in (None, "", "null") else ""
                    placeholder = "Info not found — Enter manually" if not default_value else ""

                    form_values[field_key] = st.text_input(
                        label, value=default_value, placeholder=placeholder,
                        key=f"input_{field_key}", help=field_desc,
                    )

        if st.button("💾 Save as Excel"):
            from openpyxl import Workbook
            from io import BytesIO

            wb = Workbook()
            ws = wb.active
            headers = list(form_values.keys())
            ws.append(headers)
            ws.append([form_values[h] for h in headers])

            buffer = BytesIO()
            wb.save(buffer)
            buffer.seek(0)

            st.download_button(
                "⬇️ Download Excel", data=buffer, file_name="tender_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with right:
        st.subheader("Preview")
        render_pdf_preview(st.session_state["pdf_path"])

        linked_paths = st.session_state.get("linked_paths", [])
        linked_summaries = st.session_state.get("linked_summaries", {})
        if linked_paths:
            st.subheader("📎 Linked Documents")
            for path in linked_paths:
                fname = os.path.basename(path)
                summary = linked_summaries.get(path, "")
                with st.container(border=True):
                    st.markdown(f"**{fname}**")
                    if summary:
                        st.caption(summary)
                    with open(path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download", data=f.read(), file_name=fname,
                            mime="application/pdf", key=f"download_{fname}"
                        )

else:
    st.info("Upload a PDF to get started.")