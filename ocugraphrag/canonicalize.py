from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from transformers import AutoTokenizer

from ocugraphrag.schema import Chunk


def load_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def extract_units(text: str, doc_id: str, section_path: str) -> List[dict]:
    blocks = re.split(r"\n{2,}", text.strip())
    units = []
    for idx, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        units.append({
            "doc_id": doc_id,
            "section_path": section_path,
            "text": block,
            "recommendation_strength": "",
            "evidence_level": "",
            "unit_index": idx,
        })
    return units


def chunk_text(
    text: str,
    tokenizer: AutoTokenizer,
    chunk_length: int = 512,
    overlap: int = 64,
) -> List[Tuple[int, int, str]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_length, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text_str = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append((start, end, chunk_text_str))
        if end >= len(token_ids):
            break
        start += chunk_length - overlap
    return chunks


def canonicalize_document(
    filepath: Path,
    doc_id: str,
    tokenizer: AutoTokenizer,
    chunk_length: int = 512,
    overlap: int = 64,
) -> List[Chunk]:
    raw = filepath.read_text(encoding="utf-8")
    units = extract_units(raw, doc_id, filepath.stem)
    all_chunks = []
    global_idx = 0
    for unit in units:
        pieces = chunk_text(unit["text"], tokenizer, chunk_length, overlap)
        for start, end, txt in pieces:
            all_chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{global_idx}",
                doc_id=doc_id,
                section_path=unit["section_path"],
                text=txt,
                recommendation_strength=unit["recommendation_strength"],
                evidence_level=unit["evidence_level"],
                start_char=start,
                end_char=end,
            ))
            global_idx += 1
    return all_chunks
