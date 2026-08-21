"""Baseline-delta comparison for verification checks.

This module implements the core logic for comparing baseline and post-patch
verification results, determining transitions, and classifying changes.

The baseline-delta approach ensures that:
- Pre-existing failures do not automatically make a patch fail
- Only regressions (new failures) and worsened failures block verification
- Resolved failures are properly detected and credited
- Missing baseline evidence is handled gracefully
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from patchpilot.evidence.schema import CheckTransition
from patchpilot.verification.report import CheckReport, VerificationReport


def compute_check_identity(check: CheckReport) -> str:
    """Compute a stable identity for a verification check.

    The identity is based on method, tier, normalized command, and test node
    where available. This ensures that the same check can be matched across
    baseline and post-patch phases.

    Args:
        check: CheckReport to compute identity for

    Returns:
        Stable identity string for the check
    """
    # Normalize command by removing volatile elements (paths, timestamps, etc.)
    normalized_command = _normalize_command(check.command, check.method)

    # Build identity components
    components = [
        check.method,
        check.tier,
        normalized_command,
        check.test_node,  # Test node provides fine-grained identity for pytest
    ]

    # Filter out empty components and join
    identity_parts = [c for c in components if c]
    identity = "|".join(identity_parts)

    # Hash for stable compact identifier
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def _normalize_command(command: str, method: str) -> str:
    """Normalize a command string for identity comparison.

    Removes volatile elements like specific file paths, timestamps, etc.
    while preserving the essential structure of the command.

    Args:
        command: Original command string
        method: Verification method (pytest, ruff, etc.)

    Returns:
        Normalized command string
    """
    if method == "pytest":
        # Normalize pytest commands by removing specific test paths
        # Keep the structure but normalize the test specification
        normalized = re.sub(r'pytest\s+[^\s]+', 'pytest <test_path>', command)
        normalized = re.sub(r'python\s+-m\s+pytest\s+[^\s]+', 'python -m pytest <test_path>', normalized)
        return normalized
    elif method == "ruff":
        # Ruff commands are generally stable
        return command
    elif method in ("acceptance_probe", "structural_check"):
        # Specialized checks have stable command formats
        return command
    else:
        # Generic normalization: remove specific paths
        normalized = re.sub(r'/[^\s]+', '<path>', command)
        return normalized


def compute_failure_fingerprint(check: CheckReport) -> str:
    """Compute a stable fingerprint for a failed check.

    The fingerprint captures the essential characteristics of a failure
    to determine if two failures are equivalent (same root cause) or
    different (new or worsened failure).

    Args:
        check: Failed CheckReport to compute fingerprint for

    Returns:
        Stable fingerprint string for the failure
    """
    if check.passed:
        return ""

    summary = check.summary or {}

    # Build fingerprint from failure characteristics
    components = [
        check.failure_type or "unknown",
        summary.get("error_type") or "unknown",
        # Extract key error patterns from output
        _extract_error_pattern(summary.get("relevant_output", "")),
        # Use failed tests as part of fingerprint if available
        ",".join(sorted(summary.get("failed_tests", []))) if summary.get("failed_tests") else "",
    ]

    # Filter out empty components
    fingerprint_parts = [c for c in components if c]
    fingerprint = "|".join(fingerprint_parts)

    # Hash for stable compact identifier
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def _extract_error_pattern(output: str) -> str:
    """Extract key error pattern from output for fingerprinting.

    Args:
        output: Error output string

    Returns:
        Normalized error pattern
    """
    if not output:
        return ""

    # Extract common error patterns
    # Look for exception types
    exception_match = re.search(r'(\w+Error):', output)
    if exception_match:
        return exception_match.group(1)

    # Look for assertion patterns
    assertion_match = re.search(r'AssertionError:.*?expected\s+[\'"](.+?)[\'"]\s+to', output)
    if assertion_match:
        return f"assertion_{assertion_match.group(1)}"

    # Fallback to first line of output truncated
    lines = output.split('\n')
    first_line = lines[0] if lines else ""
    return first_line[:100] if first_line else ""


def classify_transition(
    baseline_check: CheckReport | None,
    post_patch_check: CheckReport,
) -> tuple[str, str]:
    """Classify the transition from baseline to post-patch.

    Implements deterministic transition classification:
    - FAIL → PASS: RESOLVED
    - PASS → PASS: PRESERVED
    - PASS → FAIL: REGRESSION
    - FAIL → FAIL with equivalent fingerprint: PRE_EXISTING_FAILURE
    - FAIL → FAIL with changed/expanded fingerprint: WORSENED
    - no baseline match: NEW_OR_UNCOMPARED
    - not executed: UNVERIFIED

    Args:
        baseline_check: Baseline CheckReport (None if no match)
        post_patch_check: Post-patch CheckReport

    Returns:
        Tuple of (transition_type, baseline_check_id)
    """
    baseline_check_id = baseline_check.verification_id if baseline_check else ""

    if post_patch_check.failure_type == "TIMEOUT" or (
        baseline_check is not None
        and baseline_check.failure_type == "TIMEOUT"
    ):
        return CheckTransition.UNVERIFIED.value, baseline_check_id

    # If no baseline match, classify as NEW_OR_UNCOMPARED
    if baseline_check is None:
        return CheckTransition.NEW_OR_UNCOMPARED.value, baseline_check_id

    # Compute failure fingerprints for comparison
    baseline_fingerprint = compute_failure_fingerprint(baseline_check)
    post_patch_fingerprint = compute_failure_fingerprint(post_patch_check)

    # Classify based on baseline and post-patch status
    if not baseline_check.passed and post_patch_check.passed:
        # FAIL → PASS: RESOLVED
        return CheckTransition.RESOLVED.value, baseline_check_id
    elif baseline_check.passed and post_patch_check.passed:
        # PASS → PASS: PRESERVED
        return CheckTransition.PRESERVED.value, baseline_check_id
    elif baseline_check.passed and not post_patch_check.passed:
        # PASS → FAIL: REGRESSION
        return CheckTransition.REGRESSION.value, baseline_check_id
    elif not baseline_check.passed and not post_patch_check.passed:
        # FAIL → FAIL: Check if fingerprints are equivalent
        if baseline_fingerprint and baseline_fingerprint == post_patch_fingerprint:
            # Equivalent failure: PRE_EXISTING_FAILURE
            return CheckTransition.PRE_EXISTING_FAILURE.value, baseline_check_id
        else:
            # Different failure: WORSENED
            return CheckTransition.WORSENED.value, baseline_check_id
    else:
        # Unexpected state
        return CheckTransition.UNVERIFIED.value, baseline_check_id


def match_baseline_checks(
    post_patch_checks: list[CheckReport],
    baseline_checks: list[CheckReport],
) -> dict[str, CheckReport]:
    """Match post-patch checks to their baseline counterparts.

    Uses check identity to find matching baseline checks for each post-patch check.

    Args:
        post_patch_checks: List of post-patch CheckReport objects
        baseline_checks: List of baseline CheckReport objects

    Returns:
        Dictionary mapping post-patch check verification_id to matching baseline check
    """
    # Compute identities for all baseline checks
    baseline_identities = {
        compute_check_identity(check): check
        for check in baseline_checks
    }

    # Match post-patch checks to baseline
    matches: dict[str, CheckReport] = {}
    for post_check in post_patch_checks:
        post_identity = compute_check_identity(post_check)
        if post_identity in baseline_identities:
            matches[post_check.verification_id] = baseline_identities[post_identity]

    return matches


def compute_transition_summary(
    post_patch_checks: list[CheckReport],
) -> dict[str, dict[str, Any]]:
    """Compute summary statistics for check transitions.

    Args:
        post_patch_checks: List of post-patch CheckReport objects with transitions

    Returns:
        Dictionary with transition counts by tier and overall
    """
    summary: dict[str, dict[str, Any]] = {
        "overall": {
            "resolved": 0,
            "preserved": 0,
            "regression": 0,
            "pre_existing_failure": 0,
            "worsened": 0,
            "new_or_uncompared": 0,
            "unverified": 0,
        },
        "by_tier": {
            "required": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "affected": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "optional": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
        },
    }

    for check in post_patch_checks:
        # Only process post-patch checks
        if check.phase != "post_patch":
            continue
            
        transition = check.transition or CheckTransition.UNVERIFIED.value
        tier = check.tier or "unknown"

        # Convert transition to lowercase for dictionary keys
        transition_lower = transition.lower()

        # Update overall counts
        if transition_lower in summary["overall"]:
            summary["overall"][transition_lower] += 1

        # Update tier-specific counts
        if tier in summary["by_tier"] and transition_lower in summary["by_tier"][tier]:
            summary["by_tier"][tier][transition_lower] += 1

    return summary


def apply_baseline_delta_evaluation(
    report: VerificationReport | None,
    strategy: str = "balanced",
) -> tuple[str, bool]:
    """Apply baseline-delta evaluation to determine verification status.

    Replaces absolute all-green evaluation with baseline-delta comparison:
    - Ruff and constraint audit failures always block verification
    - REQUIRED target tests must pass post-patch even if they failed in baseline
    - REGRESSION and WORSENED in REQUIRED or AFFECTED tests block VERIFIED
    - PRE_EXISTING_FAILURE in AFFECTED or OPTIONAL tests must not block verification
    - OPTIONAL regressions are handled according to the configured strategy

    Args:
        report: VerificationReport with baseline and post-patch checks
        strategy: Verification strategy (strict, balanced, focused)

    Returns:
        Tuple of (verification_status, passed) based on baseline-delta evaluation
    """
    if report is None:
        # No report available, assume verification failed
        return "FAILED", False

    # Constraint and direct acceptance failures always block the patch.
    always_blocking_methods = {
        "constraint_audit",
        "acceptance_probe",
        "structural_check",
        "ast_check",
        "mock_check",
    }
    if any(
        not check.passed and check.method in always_blocking_methods
        for check in report.checks
    ):
        return "FAILED", False

    # Repository-wide Ruff is compared with baseline. Unchanged lint debt is
    # reported but is not attributed to the patch.
    if any(
        not check.passed
        and check.method == "ruff"
        and check.transition != CheckTransition.PRE_EXISTING_FAILURE.value
        for check in report.get_post_patch_checks()
    ):
        return "FAILED", False

    # Get transition summary
    transition_summary = report.transition_summary

    # Extract counts by tier (use lowercase keys)
    required = transition_summary.get("by_tier", {}).get("required", {})
    affected = transition_summary.get("by_tier", {}).get("affected", {})
    optional = transition_summary.get("by_tier", {}).get("optional", {})

    # Check for blocking transitions in AFFECTED tier.
    affected_regression = affected.get("regression", 0)
    affected_worsened = affected.get("worsened", 0)
    affected_uncompared = affected.get("new_or_uncompared", 0)

    # Check for OPTIONAL tier issues
    optional_regression = optional.get("regression", 0)
    optional_worsened = optional.get("worsened", 0)

    required_timeout = any(
        not check.passed
        and check.failure_type == "TIMEOUT"
        and check.tier in ("required", "affected")
        for check in report.get_post_patch_checks()
    )
    if required_timeout:
        return "FAILED", False

    if (
        required.get("regression", 0) > 0
        or required.get("worsened", 0) > 0
    ):
        return "FAILED", False

    # Required tests express the task contract and must pass after the patch,
    # regardless of whether they were already failing at baseline.
    if any(
        not check.passed
        and check.method == "pytest"
        and check.tier == "required"
        for check in report.get_post_patch_checks()
    ):
        return "FAILED", False

    # Check for failures in unknown or empty tiers. These cannot be safely
    # attributed and therefore block a full verification result.
    unknown_tier_failures = 0
    for check in report.checks:
        if not check.passed and check.method == "pytest":
            tier = check.tier or ""
            if tier not in ("required", "affected", "optional"):
                unknown_tier_failures += 1

    if unknown_tier_failures > 0:
        return "FAILED", False

    # New or worsened failures in tests connected to changed code block the
    # patch. Unchanged pre-existing failures remain diagnostic only.
    affected_uncompared_failures = any(
        not check.passed
        and check.method == "pytest"
        and check.tier == "affected"
        and check.transition == CheckTransition.NEW_OR_UNCOMPARED.value
        for check in report.get_post_patch_checks()
    )
    if (
        affected_regression > 0
        or affected_worsened > 0
        or (affected_uncompared > 0 and affected_uncompared_failures)
    ):
        return "FAILED", False

    if (
        affected.get("unverified", 0) > 0
        or optional.get("unverified", 0) > 0
    ):
        return "PARTIALLY_VERIFIED", True

    # PRE_EXISTING_FAILURE in AFFECTED should not block VERIFIED
    # (it was already failing before our changes)
    # No action needed - this is allowed

    # A deterministic repository-wide regression fails strict and balanced
    # verification. Focused verification retains a partial result because it
    # explicitly limits its confidence to required and affected checks.
    if strategy in ("strict", "balanced"):
        if optional_regression > 0 or optional_worsened > 0:
            return "FAILED", False
        return "VERIFIED", True
    elif strategy == "focused":
        # FOCUSED: OPTIONAL regressions don't block if REQUIRED/AFFECTED are clean
        if optional_regression > 0 or optional_worsened > 0:
            return "PARTIALLY_VERIFIED", True
        return "VERIFIED", True
    else:
        # Default to balanced
        if optional_regression > 0 or optional_worsened > 0:
            return "PARTIALLY_VERIFIED", True
        return "VERIFIED", True
