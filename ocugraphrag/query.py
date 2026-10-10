from __future__ import annotations

from typing import List

from ocugraphrag.schema import QueryState


ANSWER_TYPES = [
    "diagnosis",
    "treatment",
    "workup",
    "contraindication",
    "follow_up",
]


def parse_query(
    query: str,
    entity_extractor=None,
) -> QueryState:
    intent = ""
    entities: List[str] = []
    constraints: List[str] = []
    answer_type = "treatment"

    if entity_extractor is not None:
        result = entity_extractor(query)
        entities = result.get("entities", [])
        constraints = result.get("constraints", [])
        intent = result.get("intent", "")
        answer_type = result.get("answer_type", "treatment")
    else:
        import re
        tokens = re.findall(r"[\w\-]+", query.lower())
        entities = tokens[:5]
        intent = " ".join(tokens[:3])

    return QueryState(
        intent=intent,
        entities=entities,
        constraints=constraints,
        answer_type=answer_type,
    )
