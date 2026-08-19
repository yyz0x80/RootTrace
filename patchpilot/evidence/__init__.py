"""Evidence mapping module for linking acceptance criteria to verification results.

This module provides functionality for mapping acceptance criteria to concrete
evidence from code changes and verification results, enabling determination
of whether each criterion has been satisfied.
"""

from patchpilot.evidence.mapper import map_acceptance_evidence
from patchpilot.evidence.renderer import (
    render_acceptance_coverage,
    render_coverage_report,
)
from patchpilot.evidence.schema import (
    AcceptanceCoverageReport,
    AcceptanceEvidence,
    BehaviorChangeEvidence,
    BehaviorChangeStatus,
    BehaviorPreservationEvidence,
    BehaviorPreservationStatus,
    CompletionState,
    ConstraintEvidence,
    ConstraintStatus,
    EvidenceStatus,
    StructuralContractEvidence,
    StructuralContractStatus,
)

__all__ = [
    "AcceptanceCoverageReport",
    "AcceptanceEvidence",
    "BehaviorChangeEvidence",
    "BehaviorChangeStatus",
    "BehaviorPreservationEvidence",
    "BehaviorPreservationStatus",
    "CompletionState",
    "ConstraintEvidence",
    "ConstraintStatus",
    "EvidenceStatus",
    "StructuralContractEvidence",
    "StructuralContractStatus",
    "map_acceptance_evidence",
    "render_acceptance_coverage",
    "render_coverage_report",
]
