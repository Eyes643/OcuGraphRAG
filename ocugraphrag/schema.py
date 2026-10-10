from dataclasses import dataclass
from enum import Enum
from typing import List


class NodeType(str, Enum):
    DISEASE = "disease"
    STAGE = "stage"
    SYMPTOM = "symptom"
    EXAMINATION = "examination"
    FINDING = "finding"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    CONTRAINDICATION = "contraindication"
    FOLLOW_UP = "follow_up"
    EVIDENCE = "evidence"


class RelationGroup(str, Enum):
    DESCRIPTIVE = "descriptive"
    PROCEDURAL = "procedural"
    NORMATIVE = "normative"


RELATION_GROUPS = {
    "has_stage": RelationGroup.DESCRIPTIVE,
    "indicated_by": RelationGroup.DESCRIPTIVE,
    "requires_exam": RelationGroup.PROCEDURAL,
    "supports_diagnosis": RelationGroup.PROCEDURAL,
    "recommends": RelationGroup.NORMATIVE,
    "contraindicates": RelationGroup.NORMATIVE,
    "followed_by": RelationGroup.PROCEDURAL,
    "applies_to": RelationGroup.DESCRIPTIVE,
    "supported_by": RelationGroup.DESCRIPTIVE,
}

NODE_TYPES = [t.value for t in NodeType]
RELATION_TYPES = list(RELATION_GROUPS.keys())


@dataclass
class GraphNode:
    id: str
    label: str
    name: str
    definition: str = ""
    source_pdf: str = ""
    parent_id: str = ""


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    confidence: float = 0.0


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_path: str
    text: str
    recommendation_strength: str = ""
    evidence_level: str = ""
    start_char: int = 0
    end_char: int = 0


@dataclass
class QueryState:
    intent: str
    entities: List[str]
    constraints: List[str]
    answer_type: str


@dataclass
class EvidencePair:
    chunk: Chunk
    path: List[GraphEdge]
    score: float
