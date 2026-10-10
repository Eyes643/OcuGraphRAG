from __future__ import annotations

from typing import List, Tuple

import faiss
import numpy as np

from ocugraphrag.schema import Chunk


class DenseRetriever:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        self.index: faiss.IndexFlatIP = None
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = None

    def build_index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        output = self.model.encode(texts, batch_size=64, max_length=512)
        self.embeddings = output["dense_vecs"].astype(np.float32)
        faiss.normalize_L2(self.embeddings)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        q_out = self.model.encode([query], batch_size=1, max_length=512)
        q_vec = q_out["dense_vecs"].astype(np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def encode_single(self, text: str) -> np.ndarray:
        out = self.model.encode([text], batch_size=1, max_length=512)
        vec = out["dense_vecs"].astype(np.float32)
        faiss.normalize_L2(vec)
        return vec[0]
