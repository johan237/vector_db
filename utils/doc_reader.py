import os
import fitz  # PyMuPDF
import docx
from typing import Dict, Any, Optional


def read_pdf(path: str) -> Dict[int, str]:
    """Extract text per page from a PDF."""
    doc = fitz.open(path)
    pages = {}
    for i, page in enumerate(doc, start=1):  # start=1 → human-readable page numbers
        text = page.get_text("text")
        if text.strip():
            pages[i] = text
    return pages


def read_docx(path: str) -> Dict[int, str]:
    """Extract paragraphs from DOCX as a single pseudo-page (page=1)."""
    doc = docx.Document(path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return {1: text}


def read_txt(path: str) -> Dict[int, str]:
    """Read text file as a single pseudo-page (page=1)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return {1: text}


def read_document(path: str) -> Dict[int, str]:
    """Auto-detect file type and extract page-wise text."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    elif ext == ".txt":
        return read_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def build_doc_item(
    uid: int,
    title: str,
    description: str,
    path: str,
    author: Optional[str] = None,
    tags: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a standardized document item with metadata + per-page content.
    Returns info dict ready for SchemaVectorDB.
    """
    pages = read_document(path)

    # Flatten pages into one string for global embedding
    all_content = "\n".join(pages.values())

    return {
        "uid": uid,
        "title": title,
        "description": description,
        "content": all_content,     # used for chunking
        "pages": pages,             # keep per-page mapping
        "author": author,
        "tags": tags,
        "path": path
    }
