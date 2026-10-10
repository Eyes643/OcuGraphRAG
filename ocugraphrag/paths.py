from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from ocugraphrag.schema import GraphEdge, QueryState


RELATION_WEIGHTS: Dict[str, float] = {
    "has_stage": 0.6,
    "indicated_by": 0.7,
    "requires_exam": 0.8,
    "supports_diagnosis": 0.9,
    "recommends": 1.0,
    "contraindicates": 1.0,
    "followed_by": 0.7,
    "applies_to": 0.5,
    "supported_by": 0.8,
}


def link_entities(
    query_entities: List[str],
    G: nx.DiGraph,
    embedder,
    threshold: float = 0.75,
) -> List[str]:
    node_names = {nid: G.nodes[nid].get("name", "") for nid in G.nodes()}
    linked = []
    for entity in query_entities:
        q_vec = embedder.encode_single(entity)
        best_nid = None
        best_score = -1.0
        for nid, name in node_names.items():
            if not name:
                continue
            n_vec = embedder.encode_single(name)
            score = float(np.dot(q_vec, n_vec))
            if score > best_score:
                best_score = score
                best_nid = nid
        if best_score >= threshold and best_nid is not None:
            linked.append(best_nid)
    return linked


def bfs_paths(
    G: nx.DiGraph,
    start_nodes: List[str],
    max_depth: int = 4,
    max_paths: int = 50,
) -> List[List[Tuple[str, str, str]]]:
    all_paths = []
    for start in start_nodes:
        if start not in G:
            continue
        queue = deque()
        queue.append((start, []))
        visited_per_start = {start}
        while queue and len(all_paths) < max_paths:
            current, path_so_far = queue.popleft()
            if len(path_so_far) >= max_depth:
                continue
            for _, neighbor, edata in G.edges(current, data=True):
                if neighbor in visited_per_start:
                    continue
                edge_tuple = (current, edata.get("relation", ""), neighbor)
                new_path = path_so_far + [edge_tuple]
                all_paths.append(new_path)
                if len(all_paths) >= max_paths:
                    break
                visited_per_start.add(neighbor)
                queue.append((neighbor, new_path))
    return all_paths[:max_paths]


def score_path(
    z_q: QueryState,
    path: List[Tuple[str, str, str]],
    G: nx.DiGraph,
    mu: float = 0.05,
) -> float:
    if not path:
        return 0.0
    total = 0.0
    for src, rel, tgt in path:
        w_r = RELATION_WEIGHTS.get(rel, 0.5)
        match_val = 1.0 if any(e in G.nodes[tgt].get("name", "") for e in z_q.entities) else 0.5
        conf = G.edges[src, tgt].get("confidence", 0.5) if G.has_edge(src, tgt) else 0.5
        total += w_r * match_val * conf
    avg = total / len(path)
    penalty = mu * len(path)
    return avg - penalty


def retrieve_paths(
    z_q: QueryState,
    G: nx.DiGraph,
    embedder,
    max_depth: int = 4,
    max_paths: int = 50,
    link_threshold: float = 0.75,
) -> List[Tuple[List[Tuple[str, str, str]], float]]:
    linked = link_entities(z_q.entities, G, embedder, link_threshold)
    raw_paths = bfs_paths(G, linked, max_depth, max_paths)
    scored = []
    for p in raw_paths:
        s = score_path(z_q, p, G)
        scored.append((p, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
