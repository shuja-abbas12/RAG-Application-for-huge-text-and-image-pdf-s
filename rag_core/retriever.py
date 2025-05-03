import faiss, pickle, pathlib, numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# Same encoders we used for indexing
_TEXT_MODEL  = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
_IMAGE_MODEL = SentenceTransformer("clip-ViT-B-32",  device="cpu")


class MultiModalRetriever:
    """
    Wrap two FAISS indices (text + image captions) and return
    a merged, similarity-ranked list of chunks.
    """

    def __init__(
        self,
        text_index_path: pathlib.Path,
        image_index_path: pathlib.Path,
        text_meta_path: pathlib.Path,
        image_meta_path: pathlib.Path,
    ):
        # load faiss
        self.text_index  = faiss.read_index(str(text_index_path))
        self.image_index = faiss.read_index(str(image_index_path))

        # load metadata
        self.text_meta  = pickle.load(open(text_meta_path, "rb"))
        self.image_meta = pickle.load(open(image_meta_path, "rb"))

    # ───────────────────────────────────────────────────────────────
    def _encode_query(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        text_vec  = _TEXT_MODEL.encode(
            [query], normalize_embeddings=True).astype("float32")
        img_vec   = _IMAGE_MODEL.encode(
            [query], normalize_embeddings=True).astype("float32")
        return text_vec, img_vec

    def search(
        self,
        query: str,
        k_text: int = 4,
        k_image: int = 2,
    ) -> List[Dict]:
        """
        Returns a list of retrieved chunk dicts, each with an added
        'score' field, sorted by score desc.
        """
        q_text, q_img = self._encode_query(query)

        # text search
        D_txt, I_txt = self.text_index.search(q_text, k_text)
        txt_hits = [
            {**self.text_meta[idx], "score": float(D_txt[0, rank])}
            for rank, idx in enumerate(I_txt[0])
            if idx != -1
        ]

        # image caption search
        D_img, I_img = self.image_index.search(q_img, k_image)
        img_hits = [
            {**self.image_meta[idx], "score": float(D_img[0, rank])}
            for rank, idx in enumerate(I_img[0])
            if idx != -1
        ]

        merged = txt_hits + img_hits
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged
