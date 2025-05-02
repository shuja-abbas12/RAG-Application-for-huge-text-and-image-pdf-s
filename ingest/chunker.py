from typing import List, Dict, Iterator
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# GPT-3.5 tokenizer for accurate token counting
_enc = tiktoken.get_encoding("cl100k_base")

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # ≈ 200-250 tokens in English prose
    chunk_overlap=150,
)

def _tokens(s: str) -> int:
    return len(_enc.encode(s))

def chunk_items(items: List[Dict]) -> Iterator[Dict]:
    """
    Splits long text blocks into token-safe chunks,
    updates metadata["tokens"].
    Image chunks pass through untouched.
    """
    for item in items:
        if item["type"] == "image":
            item["metadata"]["tokens"] = 0
            yield item
            continue

        for i, chunk in enumerate(_SPLITTER.split_text(item["content"])):
            new = item.copy()
            new["id"] = f"{item['id']}_c{i}"
            new["content"] = chunk
            new["metadata"] = dict(item["metadata"])
            new["metadata"]["tokens"] = _tokens(chunk)
            yield new
