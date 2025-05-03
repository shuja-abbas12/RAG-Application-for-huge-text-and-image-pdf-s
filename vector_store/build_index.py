# ─────────────────────────────────────────────────────────────
#   vector_store/build_index.py   ←  overwrite this file
# ─────────────────────────────────────────────────────────────
"""
Build FAISS indices for text + images.

    python -m vector_store.build_index \
            --chunks data/processed/chunks.jsonl \
            --outdir vector_store
"""
import argparse, json, pickle, pathlib, numpy as np
from tqdm import tqdm
import faiss

from sentence_transformers import SentenceTransformer

# Text model (384‑dim)  ───────────────────────────────────────
_TEXT_MODEL_NAME  = "all-MiniLM-L6-v2"
_text_model       = SentenceTransformer(_TEXT_MODEL_NAME, device="cpu")

# CLIP text‑encoder arm (512‑dim)  ────────────────────────────
_IMAGE_MODEL_NAME = "clip-ViT-B-32"
_image_model      = SentenceTransformer(_IMAGE_MODEL_NAME, device="cpu")
_clip_tok         = _image_model.tokenizer      # HF tokenizer
_CLIP_MAX         = 77                          # hard limit

# -------------------------------------------------------------
def _clip_safe(text: str) -> str:
    """Return *text* truncated so that CLIP encodes ≤77 tokens."""
    ids = _clip_tok(
        text,
        truncation = True,
        max_length = _CLIP_MAX,
        add_special_tokens = True,
        return_tensors = "np",
    )["input_ids"][0]       # shape (<=77,)
    return _clip_tok.decode(ids, skip_special_tokens=True)

def embed_text(batch):                    # List[str] → (B,384)
    return _text_model.encode(
        batch, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

def embed_image_text(batch):              # List[str] → (B,512)
    safe = [_clip_safe(t) for t in batch]
    return _image_model.encode(
        safe, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

# -------------------------------------------------------------
def load_chunks(path):
    with open(path, encoding="utf8") as fh:
        for line in fh:
            yield json.loads(line)

def build_faiss(vectors):
    if not vectors:
        return None
    mat = np.stack(vectors).astype("float32")
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index, mat

# -------------------------------------------------------------
def main(chunks_path: pathlib.Path, out_dir: pathlib.Path, save_npy: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)

    text_vecs, text_meta = [], []
    img_vecs,  img_meta  = [], []

    BATCH = 512
    batch_txt, pend_txt_meta = [], []
    batch_img, pend_img_meta = [], []

    # flush helpers ----------------------------------------------------------
    def flush_text():
        nonlocal batch_txt, pend_txt_meta
        if batch_txt:
            text_vecs.extend(embed_text(batch_txt))
            text_meta.extend(pend_txt_meta)
            batch_txt, pend_txt_meta = [], []

    def flush_img():
        nonlocal batch_img, pend_img_meta
        if batch_img:
            img_vecs.extend(embed_image_text(batch_img))
            img_meta.extend(pend_img_meta)
            batch_img, pend_img_meta = [], []

    # iterate chunks ---------------------------------------------------------
    for ch in tqdm(load_chunks(chunks_path), desc="Embedding chunks"):
        if ch["type"] == "text":
            batch_txt.append(ch["content"])
            pend_txt_meta.append(ch)
            if len(batch_txt) == BATCH:
                flush_text()
        else:  # image
            caption = ch["content"] or ch["doc_id"]
            batch_img.append(caption)
            pend_img_meta.append(ch)
            if len(batch_img) == BATCH:
                flush_img()

    flush_text(); flush_img()

    # persist ---------------------------------------------------------------
    if text_vecs:
        idx, mat = build_faiss(text_vecs)
        faiss.write_index(idx, str(out_dir / "text_faiss.index"))
        pickle.dump(text_meta, open(out_dir / "text_meta.pkl", "wb"))
        if save_npy:
            np.save(out_dir / "text_vectors.npy", mat)
        print(f"✔ text index stored ({len(text_meta):,} vectors)")

    if img_vecs:
        idx, mat = build_faiss(img_vecs)
        faiss.write_index(idx, str(out_dir / "image_faiss.index"))
        pickle.dump(img_meta, open(out_dir / "image_meta.pkl", "wb"))
        if save_npy:
            np.save(out_dir / "image_vectors.npy", mat)
        print(f"✔ image index stored ({len(img_meta):,} vectors)")

# CLI -----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=pathlib.Path, default="data/processed/chunks.jsonl")
    ap.add_argument("--outdir", type=pathlib.Path,  default="vector_store")
    ap.add_argument("--no-npy", action="store_true")
    args = ap.parse_args()
    main(args.chunks, args.outdir, save_npy=not args.no_npy)
