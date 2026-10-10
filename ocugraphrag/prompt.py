from __future__ import annotations

import json
from typing import List

from ocugraphrag.schema import EvidencePair


OUTPUT_SCHEMA = {
    "direct_answer": "",
    "supporting_evidence": [],
    "contraindications": [],
    "follow_up": [],
    "citations": [],
}


def format_evidence(evidence: List[EvidencePair]) -> str:
    parts = []
    for i, ep in enumerate(evidence):
        path_str = " -> ".join(
            f"{e.source} --[{e.relation}]--> {e.target}" for e in ep.path
        ) if ep.path else "N/A"
        parts.append(
            f"[Evidence {i + 1}]\n"
            f"Text: {ep.chunk.text[:500]}\n"
            f"Source: {ep.chunk.doc_id} / {ep.chunk.section_path}\n"
            f"Evidence Level: {ep.chunk.evidence_level}\n"
            f"Graph Path: {path_str}\n"
            f"Score: {ep.score:.4f}"
        )
    return "\n\n".join(parts)


def build_prompt(
    query: str,
    evidence: List[EvidencePair],
) -> str:
    evidence_text = format_evidence(evidence)
    schema_str = json.dumps(OUTPUT_SCHEMA, indent=2, ensure_ascii=False)

    prompt = (
        "You are an ophthalmic guideline expert. "
        "Based on the retrieved evidence below, answer the clinical question. "
        "Your answer must follow the JSON schema provided.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved Evidence:\n{evidence_text}\n\n"
        f"Output JSON Schema:\n{schema_str}\n\n"
        "Requirements:\n"
        "1. Cite specific guideline sections in the citations field.\n"
        "2. Include contraindications if applicable.\n"
        "3. Include follow-up recommendations if applicable.\n"
        "4. If evidence is insufficient, state so explicitly.\n\n"
        "Answer:"
    )
    return prompt
