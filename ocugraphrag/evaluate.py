from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np


def accuracy(predictions: List[str], golds: List[str]) -> float:
    correct = sum(1 for p, g in zip(predictions, golds) if p.strip().lower() == g.strip().lower())
    return correct / max(len(golds), 1)


def macro_f1(predictions: List[str], golds: List[str], labels: List[str]) -> float:
    f1_scores = []
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, golds) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, golds) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, golds) if p != label and g == label)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def recall_at_k(
    retrieved_ids: List[List[str]],
    gold_ids: List[List[str]],
    k: int = 10,
) -> float:
    hits = 0
    total = 0
    for ret, gold in zip(retrieved_ids, gold_ids):
        top_k = set(ret[:k])
        gold_set = set(gold)
        if gold_set:
            hits += len(top_k & gold_set) / len(gold_set)
            total += 1
    return hits / max(total, 1)


def mrr(
    retrieved_ids: List[List[str]],
    gold_ids: List[List[str]],
) -> float:
    rr_sum = 0.0
    for ret, gold in zip(retrieved_ids, gold_ids):
        gold_set = set(gold)
        for rank, rid in enumerate(ret, 1):
            if rid in gold_set:
                rr_sum += 1.0 / rank
                break
    return rr_sum / max(len(retrieved_ids), 1)


def path_hit_rate(
    predicted_paths: List[List[str]],
    gold_paths: List[List[str]],
) -> float:
    hits = 0
    for pred, gold in zip(predicted_paths, gold_paths):
        pred_set = set(pred)
        gold_set = set(gold)
        if gold_set and pred_set & gold_set:
            hits += 1
    return hits / max(len(gold_paths), 1)


def hallucination_rate(
    answers: List[str],
    gold_evidence: List[List[str]],
) -> float:
    hallucinated = 0
    for ans, evidence in zip(answers, gold_evidence):
        evidence_text = " ".join(evidence).lower()
        ans_sentences = [s.strip() for s in ans.split(".") if s.strip()]
        unsupported = 0
        for sent in ans_sentences:
            if sent.lower() not in evidence_text:
                unsupported += 1
        if ans_sentences and unsupported / len(ans_sentences) > 0.5:
            hallucinated += 1
    return hallucinated / max(len(answers), 1)


def paired_bootstrap(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    diffs = np.array(scores_a) - np.array(scores_b)
    observed = float(np.mean(diffs))
    count = 0
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diff = float(np.mean(diffs[idx]))
        boot_diffs.append(boot_diff)
        if boot_diff <= 0:
            count += 1
    p_value = count / n_bootstrap
    boot_diffs_sorted = sorted(boot_diffs)
    ci_low = boot_diffs_sorted[int(0.025 * n_bootstrap)]
    ci_high = boot_diffs_sorted[int(0.975 * n_bootstrap)]
    return p_value, ci_low, ci_high


def evaluate_all(
    predictions: List[str],
    golds: List[str],
    retrieved_ids: List[List[str]] = None,
    gold_chunk_ids: List[List[str]] = None,
    predicted_paths: List[List[str]] = None,
    gold_paths: List[List[str]] = None,
    gold_evidence: List[List[str]] = None,
) -> Dict[str, float]:
    results = {
        "accuracy": accuracy(predictions, golds),
    }
    if retrieved_ids and gold_chunk_ids:
        results["recall_at_10"] = recall_at_k(retrieved_ids, gold_chunk_ids, 10)
        results["mrr"] = mrr(retrieved_ids, gold_chunk_ids)
    if predicted_paths and gold_paths:
        results["path_hit_rate"] = path_hit_rate(predicted_paths, gold_paths)
    if gold_evidence:
        results["hallucination_rate"] = hallucination_rate(predictions, gold_evidence)
    return results
