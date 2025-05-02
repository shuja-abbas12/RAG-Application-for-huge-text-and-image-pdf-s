import fitz
from pathlib import Path
import pytesseract
from PIL import Image
from typing import Iterator, Dict

# Point pytesseract to the binary
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _safe_save_pixmap(pix: fitz.Pixmap, img_path: Path) -> None:
    """
    Save *pix* as PNG, converting to RGB if the native colorspace is unsupported.
    """
    try:
        pix.save(img_path)                       # first attempt
    except ValueError as e:
        if "unsupported colorspace" not in str(e).lower():
            raise
        # ── convert exotic colorspaces (e.g., CMYK, Indexed) to RGB ──
        rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
        rgb_pix.save(img_path)
        rgb_pix = None   # free C-side memory
    finally:
        pix = None       # free original pixmap


def save_and_ocr_images(pdf_path: Path, out_dir: Path) -> Iterator[Dict]:
    """
    Extract raster images → PNG, OCR captions.  Yields dicts ready for chunker.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_id = pdf_path.stem.lower().replace(" ", "_")

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            for img_idx, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                base_name = f"{doc_id}_p{page_num:02d}_img{img_idx}.png"
                img_path = out_dir / base_name

                # Render pixmap and save safely
                pix = fitz.Pixmap(doc, xref)
                _safe_save_pixmap(pix, img_path)

                # OCR (may be blank for charts without text)
                try:
                    caption = pytesseract.image_to_string(Image.open(img_path)).strip()
                except pytesseract.TesseractNotFoundError:
                    caption = ""

                yield {
                    "id": base_name[:-4],     # strip ".png"
                    "doc_id": doc_id,
                    "page": page_num,
                    "type": "image",
                    "content": caption or "<image>",
                    "metadata": {
                        "file_path": str(img_path),
                        "tokens": 0
                    },
                }
