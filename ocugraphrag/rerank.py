from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

from ocugraphrag.schema import Chunk, EvidencePair, GraphEdge


AUTHORITY_LEVELS: Dict[str, float] = {
    "I": 1.0,
    "II": 0.7,
    "III": 0.4,
}

ACTIONABLE_LABELS = {"Treatment", "Contraindication", "Drug", "Intervention"}


def authority_score(
    chunk: Chunk,
    path: List[Tuple[str, str, str]],
    G: nx.DiGraph,
) -> float:
    level = chunk.evidence_level.strip() if chunk.evidence_level else "III"
    base = AUTHORITY_LEVELS.get(level, 0.4)
    centrality_bonus = 0.05 if "core" in chunk.section_path.lower() else 0.0
    endpoint_bonus = 0.0
    if path:
        last_node = path[-1][2]
        label = G.nodes[last_node].get("label", "") if last_node in G else ""
        if label in ACTIONABLE_LABELS:
            endpoint_bonus = 0.1
    return base + centrality_bonus + endpoint_bonus


def conflict_score(
    c1: Chunk,
    c2: Chunk,
    G: nx.DiGraph,
) -> float:
    return 0.0


def fusion_score(
    s_dense: float,
    s_graph: float,
    s_auth: float,
    s_conflict: float,
    alpha: float = 0.35,
    beta: float = 0.30,
    gamma: float = 0.20,
    delta: float = 0.15,
) -> float:
    return alpha * s_dense + beta * s_graph + gamma * s_auth - delta * s_conflict


def rerank(
    dense_results: List[Tuple[Chunk, float]],
    path_results: List[Tuple[List[Tuple[str, str, str]], float]],
    G: nx.DiGraph,
    top_b: int = 10,
    alpha: float = 0.35,
    beta: float = 0.30,
    gamma: float = 0.20,
    delta: float = 0.15,
) -> List[EvidencePair]:
    candidates = []
    for (chunk, d_score), (path, g_score) in zip(dense_results, path_results):
        auth = authority_score(chunk, path, G)
        conf = 0.0
        s = fusion_score(d_score, g_score, auth, conf, alpha, beta, gamma, delta)
        edges = []
        for src, rel, tgt in path:
            edges.append(GraphEdge(source=src, target=tgt, relation=rel))
        candidates.append(EvidencePair(chunk=chunk, path=edges, score=s))

    if len(dense_results) > len(path_results):
        for chunk, d_score in dense_results[len(path_results):]:
            auth = authority_score(chunk, [], G)
            s = fusion_score(d_score, 0.0, auth, 0.0, alpha, beta, gamma, delta)
            candidates.append(EvidencePair(chunk=chunk, path=[], score=s))

    if len(path_results) > len(dense_results):
        for path, g_score in path_results[len(dense_results):]:
            edges = [GraphEdge(source=s, target=t, relation=r) for s, r, t in path]
            s = fusion_score(0.0, g_score, 0.0, 0.0, alpha, beta, gamma, delta)
            candidates.append(EvidencePair(chunk=Chunk("", "", "", ""), path=edges, score=s))

    seen = set()
    deduped = []
    for ep in candidates:
        key = ep.chunk.chunk_id
        if key and key in seen:
            continue
        seen.add(key)
        deduped.append(ep)

    deduped.sort(key=lambda x: x.score, reverse=True)
    return deduped[:top_b]


def ranking_loss(
    s_pos: float,
    s_neg: float,
    margin: float = 0.5,
) -> float:
    return max(0.0, margin - s_pos + s_neg)
