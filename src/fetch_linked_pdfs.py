"""
FETCH LINKED PDFs: Main PDF ke andar diye gaye hyperlinks dhoondo, download karo
"""
import os
import re
import fitz  # PyMuPDF
import requests
from config import PDF_PATH

LINKED_PDFS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "linked_pdfs")


def extract_links_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_links = set()
    for page_num, page in enumerate(doc):
        for link in page.get_links():
            uri = link.get("uri")
            if uri:
                all_links.add((page_num, uri))
    doc.close()
    return sorted(all_links)


def looks_like_pdf_link(uri):
    return uri.lower().endswith(".pdf")


def is_actually_pdf(url, timeout=10):
    """Extension se dhoka mil sakta hai — asli content-type server se HEAD request karke check karna"""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        return "pdf" in resp.headers.get("Content-Type", "").lower()
    except requests.RequestException:
        return False


def download_pdf(url, save_dir, timeout=20):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()

        if "pdf" not in resp.headers.get("Content-Type", "").lower():
            print(f"   Skip: '{url}' PDF nahi lag raha")
            return None

        filename = re.sub(r"[^\w\-.]", "_", url.split("/")[-1].split("?")[0])
        if not filename or not filename.lower().endswith(".pdf"):
            filename = f"linked_doc_{abs(hash(url)) % 100000}.pdf"

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        print(f"   Downloaded: {filename}")
        return save_path
    except requests.RequestException as e:
        print(f"   Download failed for '{url}': {e}")
        return None


def fetch_all_linked_pdfs(pdf_path=None, save_dir=LINKED_PDFS_DIR):
    path_to_use = pdf_path or PDF_PATH
    print(f"Scanning links in: {path_to_use}\n")

    links = extract_links_from_pdf(path_to_use)
    print(f"Total links found: {len(links)}\n")

    downloaded_paths = []
    for page_num, uri in links:
        if not uri.startswith("http"):
            continue   # mailto:, internal anchors, etc. skip karo
        if looks_like_pdf_link(uri):
            path = download_pdf(uri, save_dir)
            if path:
                downloaded_paths.append(path)
        else:
            if is_actually_pdf(uri):
                path = download_pdf(uri, save_dir)
                if path:
                    downloaded_paths.append(path)

    print(f"\nTotal PDFs successfully downloaded: {len(downloaded_paths)}")
    return downloaded_paths


if __name__ == "__main__":
    fetch_all_linked_pdfs()