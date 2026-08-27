"""Evidence and hypothesis domain capability."""

from roottrace.evidence.graph import EvidenceGraph, aggregate_evidence
from roottrace.evidence.schema import (
    AgentFinding,
    AgentRole,
    ConfidenceLevel,
    EvidenceEdge,
    EvidenceItem,
    EvidenceKind,
    EvidenceRelation,
    FindingStatus,
    Hypothesis,
    HypothesisDisposition,
    SourceLocation,
    UncertaintyLevel,
    VerificationStep,
)

__all__ = [
    "AgentFinding", "AgentRole", "ConfidenceLevel", "EvidenceEdge",
    "EvidenceGraph", "EvidenceItem", "EvidenceKind", "EvidenceRelation",
    "FindingStatus", "Hypothesis", "HypothesisDisposition", "SourceLocation",
    "UncertaintyLevel", "VerificationStep", "aggregate_evidence",
]
