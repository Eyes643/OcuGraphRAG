from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ocugraphrag.schema import GraphNode, GraphEdge, Chunk


def load_graph(json_path: str | Path) -> nx.DiGraph:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.DiGraph()
    for n in data.get("nodes", []):
        G.add_node(n["id"], **n)
    for e in data.get("links", []):
        G.add_edge(e["source"], e["target"], **e)
    return G


def get_nodes(G: nx.DiGraph) -> List[GraphNode]:
    nodes = []
    for nid, attrs in G.nodes(data=True):
        nodes.append(GraphNode(
            id=nid,
            label=attrs.get("label", ""),
            name=attrs.get("name", ""),
            definition=attrs.get("definition", ""),
            source_pdf=attrs.get("source_pdf", ""),
            parent_id=attrs.get("parent_id", ""),
        ))
    return nodes


def get_edges(G: nx.DiGraph) -> List[GraphEdge]:
    edges = []
    for u, v, attrs in G.edges(data=True):
        edges.append(GraphEdge(
            source=u,
            target=v,
            relation=attrs.get("relation", ""),
            confidence=attrs.get("confidence", 0.0),
        ))
    return edges


def compute_edge_confidence(s_link: float, s_rel: float) -> float:
    return math.sqrt(max(s_link, 0.0) * max(s_rel, 0.0))


def build_alignment_matrix(
    chunks: List[Chunk],
    G: nx.DiGraph,
    embedder=None,
) -> np.ndarray:
    node_ids = list(G.nodes())
    n_chunks = len(chunks)
    n_nodes = len(node_ids)
    A = np.zeros((n_chunks, n_nodes), dtype=np.float32)
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    for ci, chunk in enumerate(chunks):
        for nid in node_ids:
            name = G.nodes[nid].get("name", "")
            if name and name in chunk.text:
                A[ci, node_id_to_idx[nid]] = 1.0
    return A


def filter_edges_by_confidence(
    G: nx.DiGraph,
    threshold: float = 0.85,
) -> nx.DiGraph:
    filtered = nx.DiGraph()
    for nid, attrs in G.nodes(data=True):
        filtered.add_node(nid, **attrs)
    for u, v, attrs in G.edges(data=True):
        if attrs.get("confidence", 1.0) >= threshold:
            filtered.add_edge(u, v, **attrs)
    return filtered
