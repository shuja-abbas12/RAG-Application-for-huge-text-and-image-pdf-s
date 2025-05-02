"""
End-to-end CLI:

    python -m ingest.pipeline \
        --pdf-dir data/raw_pdfs \
        --out-json data/processed/chunks.jsonl
"""
import argparse, json, itertools
from pathlib import Path
from tqdm import tqdm

from .extract_text import iter_pdf_text
from .extract_images import save_and_ocr_images
from .chunker import chunk_items

def build_chunks(pdf_dir: Path, out_json: Path, img_dir: Path):
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDFs found in", pdf_dir)

    with out_json.open("w", encoding="utf8") as fh:
        for pdf in tqdm(pdf_files, desc="PDFs"):
            text_iter = iter_pdf_text(pdf)
            img_iter = save_and_ocr_images(pdf, img_dir)
            merged = itertools.chain(text_iter, img_iter)
            for chunk in chunk_items(list(merged)):
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path, default=Path("data/raw_pdfs"))
    ap.add_argument("--out-json", type=Path, default=Path("data/processed/chunks.jsonl"))
    ap.add_argument("--img-dir", type=Path, default=Path("data/processed/images"))
    args = ap.parse_args()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.img_dir.mkdir(parents=True, exist_ok=True)

    build_chunks(args.pdf_dir, args.out_json, args.img_dir)
    print(f"\n✅  Done. Chunks saved to {args.out_json}")
