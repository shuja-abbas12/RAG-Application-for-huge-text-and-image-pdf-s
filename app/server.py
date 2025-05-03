# app/server.py
import os, pathlib, time, json
from io import BytesIO
from typing import List, Dict

# ────────────────────────────────────────────────────────────
# ❶  Load .env *before* anything imports OpenAI / Rag chain
# ────────────────────────────────────────────────────────────
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())          # makes OPENAI_API_KEY visible

# ── 3rd‑party & std‑lib imports that *use* the env afterwards ──
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image, UnidentifiedImageError
import pytesseract
import numpy as np, pickle, umap
from plotly.utils import PlotlyJSONEncoder

from rag_core import RagQAChain      # ← OpenAI instantiated here, env is ready
# ────────────────────────────────────────────────────────────

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

app = FastAPI(title="Multimodal RAG Demo")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

rag = RagQAChain()
telemetry: List[Dict] = []

# ========== ROUTES =========================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ask")
async def ask(
    query: str = Form(None),
    image: UploadFile = File(None),
    k_text: int = Form(4),
    k_img: int = Form(2),
):
    # ── optional OCR on uploaded image ──
    if image and image.filename:
        try:
            raw = await image.read()
            img = Image.open(BytesIO(raw))
            ocr_text = pytesseract.image_to_string(img).strip()
            query = f"{query or ''} {ocr_text}".strip()
        except UnidentifiedImageError:
            return JSONResponse(
                {"error": "Uploaded file is not a valid image."}, status_code=400
            )

    if not query:
        return JSONResponse({"error": "Empty query."}, status_code=400)

    t0 = time.time()
    answer, sources = rag.ask(query, k_text, k_img)
    latency = time.time() - t0

    telemetry.append(
        dict(query=query, latency=latency,
             k_text=k_text, k_img=k_img, n_sources=len(sources))
    )

    return JSONResponse({"answer": answer,
                         "sources": sources,
                         "latency": latency})


@app.get("/api/umap")
async def umap_data():
    vec_dir = pathlib.Path("vector_store")
    txt_vecs = np.load(vec_dir / "text_vectors.npy")
    meta     = pickle.load(open(vec_dir / "text_meta.pkl", "rb"))
    labels   = [m["doc_id"].split("_")[0].upper() for m in meta]

    reducer  = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    coords   = reducer.fit_transform(txt_vecs)

    return {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "labels": labels,
    }


@app.get("/api/metrics")
async def metrics():
    return JSONResponse(telemetry)
