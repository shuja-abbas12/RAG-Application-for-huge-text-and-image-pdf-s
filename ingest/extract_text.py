import fitz  # PyMuPDF
from pathlib import Path
from typing import Iterator, Dict


def _block_to_text(block: dict) -> str:
    """
    Robustly flatten a text block coming from page.get_text("dict").
    Works for both PyMuPDF ≤1.23 (lines[*]["text"]) and ≥1.24 (lines[*]["spans"]).
    """
    pieces = []
    for line in block.get("lines", []):
        # Newer PyMuPDF: text only inside spans
        if "spans" in line:
            pieces.extend(span["text"] for span in line["spans"])
        # Fallback for very old versions
        elif "text" in line:
            pieces.append(line["text"])
    return " ".join(pieces).strip()


def iter_pdf_text(pdf_path: Path, min_chars: int = 30) -> Iterator[Dict]:
    """
    Yields paragraph-level chunks with metadata.
    Tiny noise blocks (< min_chars) are skipped.
    """
    doc_id = pdf_path.stem.lower().replace(" ", "_")

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            for b_idx, block in enumerate(page.get_text("dict")["blocks"]):
                if block.get("type", 1) != 0:        # keep text blocks only
                    continue
                text = _block_to_text(block)
                if len(text) < min_chars:
                    continue

                yield {
                    "id": f"{doc_id}_p{page_num:02d}_t{b_idx}",
                    "doc_id": doc_id,
                    "page": page_num,
                    "type": "text",
                    "content": text,
                    "metadata": {
                        "tokens": None  # filled later by chunker
                    },
                }
