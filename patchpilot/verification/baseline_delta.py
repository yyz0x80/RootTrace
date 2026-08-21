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


def compute_failure_fingerprint(check: CheckReport) -> str | dict[str, str]:
    """Compute a stable fingerprint for a failed check.

    For pytest checks, returns a mapping of test nodes to their specific error fingerprints.
    For other checks, returns a single fingerprint string for backward compatibility.

    The fingerprint captures the essential characteristics of a failure
    to determine if two failures are equivalent (same root cause) or
    different (new or worsened failure).

    Args:
        check: Failed CheckReport to compute fingerprint for

    Returns:
        For pytest: Dict mapping test nodes to error fingerprints
        For other methods: Stable fingerprint string for the failure
    """
    if check.passed:
        return ""

    # For pytest, generate per-test failure mappings
    if check.method == "pytest":
        return _compute_pytest_failure_mapping(check)

    # For other methods, use the original single fingerprint approach
    summary = check.summary or {}
    failure_detail = (
        summary.get("error")
        or summary.get("message")
        or summary.get("relevant_output")
        or summary.get("output")
        or ""
    )

    # Build fingerprint from failure characteristics
    components = [
        check.failure_type or "unknown",
        summary.get("error_type") or "unknown",
        # Extract key error patterns from output
        _extract_error_pattern(str(failure_detail)),
        _normalize_failure_detail(str(failure_detail)),
        # Use failed tests as part of fingerprint if available
        ",".join(sorted(summary.get("failed_tests", []))) if summary.get("failed_tests") else "",
    ]

    # Filter out empty components
    fingerprint_parts = [c for c in components if c]
    fingerprint = "|".join(fingerprint_parts)

    # Hash for stable compact identifier
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def _normalize_failure_detail(detail: str) -> str:
    """Normalize bounded error text for stable failure comparison."""
    return " ".join(detail.split())[:500]


def _compute_pytest_failure_mapping(check: CheckReport) -> dict[str, str]:
    """Compute per-test failure mapping for pytest checks.

    Parses pytest output to extract individual test failures and their
    specific error fingerprints, enabling granular comparison of which
    tests failed and why.

    Uses full output parsing by failure sections to avoid truncation issues
    that can occur with limited relevant_output.

    Args:
        check: Failed pytest CheckReport to compute failure mapping for

    Returns:
        Dict mapping test node IDs to their error fingerprints
    """
    summary = check.summary or {}
    # Try to get full output from summary if available, otherwise use relevant_output
    full_output = summary.get("full_output", summary.get("relevant_output", ""))
    failed_tests = summary.get("failed_tests", [])

    failure_mapping: dict[str, str] = {}

    # If we have failed tests list, parse output for each test's error
    if failed_tests:
        for test_node in failed_tests:
            # Extract error specific to this test from full output
            test_error = _extract_test_error_section(full_output, test_node)
            error_fingerprint = _compute_error_fingerprint(test_error, check.failure_type)
            failure_mapping[test_node] = error_fingerprint

    return failure_mapping


def _extract_test_error_section(output: str, test_node: str) -> str:
    """Extract error output section specific to a test node from full output.

    Parses the full pytest output to find the complete error section for a specific test,
    avoiding truncation issues that can occur with limited relevant_output.

    Args:
        output: Full pytest output
        test_node: Test node identifier to extract error for

    Returns:
        Complete error output section for the test node
    """
    if not output:
        return ""

    lines = output.split('\n')

    test_error_lines = []

    for i, line in enumerate(lines):
        if f"FAILED {test_node}" in line:
            test_error_lines.append(line)
            # Include subsequent lines until next test failure section or end
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                # Stop if we hit another test's failure section
                if "FAILED " in next_line and test_node not in next_line:
                    break
                test_error_lines.append(next_line)
            break

    return "\n".join(test_error_lines) if test_error_lines else ""


def _compute_error_fingerprint(error_output: str, failure_type: str | None) -> str:
    """Compute fingerprint for a specific error.

    Includes error message content for all exception types to detect when
    the same test fails with different error details, not just different error types.

    Args:
        error_output: Error output for a specific test
        failure_type: Overall failure type from the check

    Returns:
        Stable fingerprint for the error
    """
    components = [
        failure_type or "unknown",
        _extract_error_pattern(error_output),
    ]

    # Extract error message content for all exception types for better comparison
    if failure_type:
        # Try to extract the error message after the exception type
        error_message_match = re.search(rf'{failure_type}:\s*(.+)', error_output)
        if error_message_match:
            # Take first 50 chars of error message for stability
            error_message = error_message_match.group(1)[:50].replace(" ", "_").replace("\n", "_")
            components.append(error_message)
        else:
            # Fallback: extract any message-like content
            message_match = re.search(r':\s*(.+)', error_output)
            if message_match:
                error_message = message_match.group(1)[:50].replace(" ", "_").replace("\n", "_")
                components.append(error_message)

    # Filter out empty components
    fingerprint_parts = [c for c in components if c]
    fingerprint = "|".join(fingerprint_parts)

    # Hash for stable compact identifier
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def _compute_overall_fingerprint(check: CheckReport) -> str:
    """Compute overall fingerprint for a check when per-test comparison is not possible.

    Used as fallback for collection errors, import errors, and other cases where
    individual test failure mapping is not available.

    Returns empty string if insufficient information for reliable fingerprint.

    Args:
        check: CheckReport to compute overall fingerprint for

    Returns:
        Stable fingerprint for the overall failure, or empty string if unreliable
    """
    if check.passed:
        return ""

    summary = check.summary or {}

    # Check if we have sufficient information for reliable fingerprint
    has_failure_type = check.failure_type is not None and check.failure_type not in (None, "unknown")
    has_error_type = summary.get("error_type") is not None and summary.get("error_type") not in (None, "unknown")
    has_output = bool(summary.get("full_output", summary.get("relevant_output", "")))
    has_failed_tests = bool(summary.get("failed_tests"))

    # If we lack basic failure information, return empty to indicate unreliability
    if not (has_failure_type or has_error_type or has_output or has_failed_tests):
        return ""

    # Build fingerprint from overall failure characteristics
    components = [
        check.failure_type or "unknown",
        summary.get("error_type") or "unknown",
        # Use key error patterns from full output if available
        _extract_error_pattern(summary.get("full_output", summary.get("relevant_output", ""))),
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

    Implements deterministic transition classification with per-test failure analysis:
    - FAIL → PASS: RESOLVED
    - PASS → PASS: PRESERVED
    - PASS → FAIL: REGRESSION
    - FAIL → FAIL with equivalent fingerprint: PRE_EXISTING_FAILURE
    - FAIL → FAIL with changed/expanded fingerprint: WORSENED
    - Some failures resolved without new regressions: IMPROVED
    - no baseline match: NEW_OR_UNCOMPARED
    - not executed: UNVERIFIED

    For pytest checks, performs granular per-test comparison to detect IMPROVED state
    where some historical failures are resolved without introducing new failures.

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

    # For pytest, use granular per-test comparison
    if post_patch_check.method == "pytest":
        return _classify_pytest_transition(baseline_check, post_patch_check, baseline_check_id)

    # For other methods, use original fingerprint comparison
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


def _classify_pytest_transition(
    baseline_check: CheckReport,
    post_patch_check: CheckReport,
    baseline_check_id: str,
) -> tuple[str, str]:
    """Classify pytest transition using per-test failure analysis.

    Compares baseline and post-patch test failures at the individual test level
    to detect nuanced transitions like IMPROVED (some failures resolved without
    new regressions).

    Args:
        baseline_check: Baseline pytest CheckReport
        post_patch_check: Post-patch pytest CheckReport
        baseline_check_id: Verification ID of baseline check

    Returns:
        Tuple of (transition_type, baseline_check_id)
    """
    # Get failure mappings for both checks
    baseline_failures = compute_failure_fingerprint(baseline_check)
    post_patch_failures = compute_failure_fingerprint(post_patch_check)

    # Handle backward compatibility if fingerprints are strings
    if isinstance(baseline_failures, str) or isinstance(post_patch_failures, str):
        # Fall back to original comparison
        if not baseline_check.passed and post_patch_check.passed:
            return CheckTransition.RESOLVED.value, baseline_check_id
        elif baseline_check.passed and post_patch_check.passed:
            return CheckTransition.PRESERVED.value, baseline_check_id
        elif baseline_check.passed and not post_patch_check.passed:
            return CheckTransition.REGRESSION.value, baseline_check_id
        elif not baseline_check.passed and not post_patch_check.passed:
            if baseline_failures == post_patch_failures:
                return CheckTransition.PRE_EXISTING_FAILURE.value, baseline_check_id
            else:
                return CheckTransition.WORSENED.value, baseline_check_id

    # Ensure we have dict mappings
    baseline_map = baseline_failures if isinstance(baseline_failures, dict) else {}
    post_patch_map = post_patch_failures if isinstance(post_patch_failures, dict) else {}

    # Classify based on overall status first
    if not baseline_check.passed and post_patch_check.passed:
        # All tests now pass: RESOLVED
        return CheckTransition.RESOLVED.value, baseline_check_id
    elif baseline_check.passed and post_patch_check.passed:
        # All tests passed in both: PRESERVED
        return CheckTransition.PRESERVED.value, baseline_check_id
    elif baseline_check.passed and not post_patch_check.passed:
        # New failures introduced: REGRESSION
        return CheckTransition.REGRESSION.value, baseline_check_id
    elif not baseline_check.passed and not post_patch_check.passed:
        # Both have failures - analyze per-test transitions
        return _analyze_pytest_failure_transition(
            baseline_map,
            post_patch_map,
            baseline_check_id,
            baseline_check,
            post_patch_check
        )
    else:
        return CheckTransition.UNVERIFIED.value, baseline_check_id


def _analyze_pytest_failure_transition(
    baseline_map: dict[str, str],
    post_patch_map: dict[str, str],
    baseline_check_id: str,
    baseline_check: CheckReport,
    post_patch_check: CheckReport,
) -> tuple[str, str]:
    """Analyze pytest failure transitions at per-test level.

    Determines if the transition is IMPROVED, PRE_EXISTING_FAILURE, or WORSENED
    by comparing individual test failures and their error fingerprints.

    When failure maps are empty (e.g., collection errors, import errors),
    falls back to overall fingerprint comparison to provide more accurate classification.

    Args:
        baseline_map: Mapping of test nodes to error fingerprints in baseline
        post_patch_map: Mapping of test nodes to error fingerprints in post-patch
        baseline_check_id: Verification ID of baseline check
        baseline_check: Baseline CheckReport for overall fingerprint comparison
        post_patch_check: Post-patch CheckReport for overall fingerprint comparison

    Returns:
        Tuple of (transition_type, baseline_check_id)
    """
    # Handle empty failure maps (collection errors, import errors, etc.)
    if not baseline_map and not post_patch_map:
        # Both maps empty - use overall fingerprint comparison
        baseline_fingerprint = _compute_overall_fingerprint(baseline_check)
        post_patch_fingerprint = _compute_overall_fingerprint(post_patch_check)

        if baseline_fingerprint and baseline_fingerprint == post_patch_fingerprint:
            # Same overall error: PRE_EXISTING_FAILURE
            return CheckTransition.PRE_EXISTING_FAILURE.value, baseline_check_id
        elif baseline_fingerprint and post_patch_fingerprint:
            # Different overall error: WORSENED
            return CheckTransition.WORSENED.value, baseline_check_id
        else:
            # Cannot reliably generate fingerprint: UNVERIFIED
            return CheckTransition.UNVERIFIED.value, baseline_check_id

    if baseline_map and not post_patch_map:
        post_patch_fingerprint = _compute_overall_fingerprint(post_patch_check)
        if post_patch_fingerprint:
            return CheckTransition.WORSENED.value, baseline_check_id
        return CheckTransition.UNVERIFIED.value, baseline_check_id

    if not baseline_map and post_patch_map:
        return CheckTransition.REGRESSION.value, baseline_check_id

    resolved_tests = []
    regressed_tests = []
    pre_existing_tests = []
    worsened_tests = []

    # Check each baseline test
    for test_node, baseline_fingerprint in baseline_map.items():
        if test_node not in post_patch_map:
            # Test no longer fails: RESOLVED
            resolved_tests.append(test_node)
        elif post_patch_map[test_node] == baseline_fingerprint:
            # Same test, same error: PRE_EXISTING_FAILURE
            pre_existing_tests.append(test_node)
        else:
            # Same test, different error: WORSENED
            worsened_tests.append(test_node)

    # Check for new failures (regressions)
    for test_node in post_patch_map:
        if test_node not in baseline_map:
            # New test failure: REGRESSION
            regressed_tests.append(test_node)

    # Determine overall transition based on per-test analysis
    if regressed_tests:
        # New failures introduced: REGRESSION
        return CheckTransition.REGRESSION.value, baseline_check_id
    elif worsened_tests:
        # Existing failures worsened: WORSENED
        return CheckTransition.WORSENED.value, baseline_check_id
    elif resolved_tests and not pre_existing_tests:
        # All failures resolved: RESOLVED
        return CheckTransition.RESOLVED.value, baseline_check_id
    elif resolved_tests and pre_existing_tests:
        # Some failures resolved, some remain: IMPROVED
        return CheckTransition.IMPROVED.value, baseline_check_id
    elif pre_existing_tests and not resolved_tests:
        # Failures unchanged: PRE_EXISTING_FAILURE
        return CheckTransition.PRE_EXISTING_FAILURE.value, baseline_check_id
    else:
        # No failures in either (shouldn't reach here given outer condition)
        # Return NEW_OR_UNCOMPARED instead of PRESERVED for safety
        return CheckTransition.NEW_OR_UNCOMPARED.value, baseline_check_id


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
            "improved": 0,
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
                "improved": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "affected": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "improved": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "optional": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "improved": 0,
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
    # IMPROVED is non-blocking for AFFECTED tier (similar to PRE_EXISTING_FAILURE)

    # Check for OPTIONAL tier issues
    optional_regression = optional.get("regression", 0)
    optional_worsened = optional.get("worsened", 0)
    # IMPROVED is non-blocking for OPTIONAL tier (similar to PRE_EXISTING_FAILURE)

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
    # IMPROVED transition does not apply to required tests - they must pass completely.
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

    # NEW_OR_UNCOMPARED failures in optional tier (e.g., collection errors)
    # These indicate unreliable comparison and should not result in VERIFIED
    optional_uncompared_failures = any(
        not check.passed
        and check.method == "pytest"
        and check.tier == "optional"
        and check.transition == CheckTransition.NEW_OR_UNCOMPARED.value
        for check in report.get_post_patch_checks()
    )
    if optional_uncompared_failures:
        # For strict and balanced strategies, NEW_OR_UNCOMPARED should block VERIFIED
        if strategy in ("strict", "balanced"):
            return "FAILED", False
        # For focused strategy, return PARTIALLY_VERIFIED
        elif strategy == "focused":
            return "PARTIALLY_VERIFIED", True

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
