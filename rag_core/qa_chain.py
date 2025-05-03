"""
High-level ask() call:
    from rag_core import RagQAChain
    qa = RagQAChain()
    answer, sources = qa.ask("Your question?")
"""
import pathlib, yaml, datetime
from typing import Tuple, List, Dict
from openai import OpenAI
from .retriever import MultiModalRetriever
from .prompt_templates import build_prompt

# ── tiny config block you might tweak ──────────────────────────────
_VECTOR_DIR   = pathlib.Path("vector_store")
_OPENAI_MODEL = "gpt-4o-mini"        # or "gpt-4o", "gpt-3.5-turbo"
_LOG_PATH     = pathlib.Path("prompt_logs.yaml")
# ──────────────────────────────────────────────────────────────────

client = OpenAI()   # uses OPENAI_API_KEY env var

class RagQAChain:
    def __init__(self):
        self.retriever = MultiModalRetriever(
            _VECTOR_DIR / "text_faiss.index",
            _VECTOR_DIR / "image_faiss.index",
            _VECTOR_DIR / "text_meta.pkl",
            _VECTOR_DIR / "image_meta.pkl",
        )

    # --------------------------------------------------------------
    def _call_llm(self, prompt: str) -> str:
        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()

    # --------------------------------------------------------------
    def ask(self, question: str, k_text: int = 4, k_image: int = 2
            ) -> Tuple[str, List[Dict]]:
        hits = self.retriever.search(question, k_text, k_image)
        prompt = build_prompt(question, hits)

        answer = self._call_llm(prompt)

        # simple source extraction: any token like "(id)"
        cited_ids = {cid for cid in (h["id"] for h in hits) if cid in answer}

        sources = [h for h in hits if h["id"] in cited_ids]

        # ---------- prompt log (for report reproducibility) -----------
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "question": question,
            "prompt": prompt,
            "answer": answer,
            "source_ids": cited_ids,
        }
        with open(_LOG_PATH, "a", encoding="utf8") as f:
            yaml.safe_dump([log_entry], f, sort_keys=False)

        return answer, sources
