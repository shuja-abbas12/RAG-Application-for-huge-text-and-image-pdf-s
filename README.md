# Multimodal RAG for FAST-NUCES Documents

This repository implements a **Multimodal Retrieval-Augmented Generation (RAG)** system over three FAST-NUCES PDFs (Annual Report, PwC Financials slides, FYP Handbook). It supports both text and image‐based OCR retrieval, semantic search via FAISS, and answer generation with an LLM.

---

## 🚀 Features

- **Text & Image Ingestion**  
  - Layout-aware PDF text extraction via PyMuPDF  
  - OCR of embedded images via Tesseract
- **Chunking & Embedding**  
  - Token-safe text chunks (LangChain text splitter + tiktoken)  
  - CLIP/BLIP embeddings for image captions  
  - Sentence-Transformer embeddings for text
- **Vector Store**  
  - FAISS indices for fast similarity search  
  - Saved `.npy` + `.pkl` metadata for reproducibility
- **RAG Layer**  
  - LangChain retriever over combined text+image indices  
  - OpenAI LLM (GPT-4o mini) for answer generation
- **Web Interface**  
  - FastAPI back end + Jinja2 templates  
  - Live answer streaming, query + image uploads  
  - `/api/umap` endpoint for embedding UMAP

---

## 📦 Prerequisites

- Python 3.10+
- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) installed & in your `PATH`
- An OpenAI API key
- (Optional) CUDA GPU for faster embedding/model training

---

## 🔧 Installation

1. **Clone the repo**
   git clone https://github.com/shuja-abbas12/GEN-A3.git
   cd GEN-A3

2. **Create & activate a virtual environment**
   python -m venv .rag
   source .rag/bin/activate    # Linux / macOS
   .rag\Scripts\activate       # Windows PowerShell

3. **Install dependencies**
   pip install --upgrade pip
   pip install -r requirements.txt

## 🔑 Environment Variables

   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o-mini

## 🗂️ Data Ingestion & Chunking

python -m ingest.pipeline \
  --pdf-dir data/raw_pdfs \
  --out-json data/processed/chunks.jsonl \
  --img-dir data/processed/images
## 🔢 Build Vector Store

python -m vector_store.build_index \
  --chunks data/processed/chunks.jsonl \
  --outdir vector_store

## 🚀 Run the FastAPI Server

uvicorn app.server:app --reload --host 127.0.0.1 --port 8000

## 📝 Project Structure

.
├── .env
├── ingest/
│   ├── extract_text.py
│   ├── extract_images.py
│   ├── chunker.py
│   └── pipeline.py
├── vector_store/
│   ├── build_index.py
│   └── (output files: *.faiss, *.npy, *.pkl)
├── app/
│   ├── server.py
│   ├── templates/
│   └── static/
├── rag_core/
│   └── qa_chain.py   # your RAG wrapper
├── data/
│   └── raw_pdfs/     # put your PDFs here
├── requirements.txt
└── README.md         # ← this file

