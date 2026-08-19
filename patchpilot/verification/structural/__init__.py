"""Structural Checker module for PatchPilot verification.

This module provides AST-based structural verification to ensure code changes
meet structural requirements without executing the code.

Supported checks:
- Function or method existence
- Signature preservation
- Call relationship verification
- Import restriction enforcement
- Class method/decorator existence
"""

from patchpilot.verification.structural.ast_checks import (
    ASTChecker,
    CheckResult,
    CheckType,
    StructuralCheck,
)
from patchpilot.verification.structural.runner import (
    StructuralReport,
    StructuralRunner,
)

__all__ = [
    "ASTChecker",
    "CheckResult",
    "CheckType",
    "StructuralCheck",
    "StructuralReport",
    "StructuralRunner",
]
