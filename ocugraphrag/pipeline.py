from __future__ import annotations

from typing import Dict, List, Optional

import yaml
import networkx as nx

from ocugraphrag.schema import Chunk, EvidencePair, QueryState
from ocugraphrag.query import parse_query
from ocugraphrag.dense import DenseRetriever
from ocugraphrag.graph import load_graph
from ocugraphrag.paths import retrieve_paths
from ocugraphrag.rerank import rerank
from ocugraphrag.prompt import build_prompt
from ocugraphrag.generate import Generator


class OcuGraphRAGPipeline:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.dense_retriever: Optional[DenseRetriever] = None
        self.G: Optional[nx.DiGraph] = None
        self.generator: Optional[Generator] = None
        self.chunks: List[Chunk] = []

    def load_graph(self, graph_path: str):
        self.G = load_graph(graph_path)

    def build_dense_index(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.dense_retriever = DenseRetriever(self.cfg["embedding"]["model"])
        self.dense_retriever.build_index(chunks)

    def load_generator(self, adapter_path: Optional[str] = None):
        self.generator = Generator(
            base_model=self.cfg["model"]["base"],
            adapter_path=adapter_path,
            max_context=self.cfg["model"]["max_context"],
            max_new_tokens=self.cfg["model"]["max_new_tokens"],
        )

    def infer(self, query: str) -> str:
        z_q = parse_query(query)

        dense_results = self.dense_retriever.search(
            query, self.cfg["retrieval"]["dense_top_k"]
        )

        path_results = retrieve_paths(
            z_q, self.G, self.dense_retriever,
            max_depth=self.cfg["graph"]["bfs_max_depth"],
            max_paths=self.cfg["graph"]["max_paths_per_query"],
            link_threshold=self.cfg["graph"]["entity_link_threshold"],
        )

        rcfg = self.cfg["rerank"]
        evidence = rerank(
            dense_results, path_results, self.G,
            top_b=self.cfg["retrieval"]["top_b"],
            alpha=rcfg["alpha"],
            beta=rcfg["beta"],
            gamma=rcfg["gamma"],
            delta=rcfg["delta"],
        )

        prompt = build_prompt(query, evidence)
        answer = self.generator.generate(prompt)
        return answer
