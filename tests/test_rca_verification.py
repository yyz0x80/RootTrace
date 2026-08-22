"""Tests for runtime test verification of RCA hypotheses."""

from __future__ import annotations

from pathlib import Path

from patchpilot.rca.context_builder import capture_repository_fingerprint
from patchpilot.rca.sandbox import RuntimeVerificationSandbox
from patchpilot.rca.schema import (
    AgentRole,
    EvidenceGraph,
    EvidenceItem,
    EvidenceKind,
    Hypothesis,
    HypothesisDisposition,
    IncidentInput,
    Provenance,
    VerificationOutcome,
    VerificationStatus,
    VerificationStep,
)
from patchpilot.rca.verification import RuntimeTestVerifier


def _write(repo: Path, relative_path: str, content: str) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _graph(git_repo, hypotheses: list[Hypothesis]) -> EvidenceGraph:
    incident = IncidentInput(
        id="inc-001",
        repo="target",
        base_commit=git_repo.base_sha,
        title="multiply returns a+b",
        problem="multiply returns a+b instead of a*b",
        logs=[],
        provenance=Provenance(source="test_fixture"),
    )
    seed = EvidenceItem(
        id="ev-seed-001",
        agent=AgentRole.CODE,
        kind=EvidenceKind.CODE_SNIPPET,
        observation="suspicious function",
        provenance=Provenance(source="test_fixture"),
        excerpt="def multiply(a, b):\n    return a + b",
    )
    return EvidenceGraph(
        incident=incident,
        findings=[],
        evidence=[seed],
        hypotheses=hypotheses,
    )


def _hypothesis(
    hypothesis_id: str,
    *commands: str,
    timeouts: list[int | None] | None = None,
) -> Hypothesis:
    timeouts = timeouts or [None] * len(commands)
    return Hypothesis(
        id=hypothesis_id,
        statement=f"{hypothesis_id} claim",
        supporting_evidence_ids=["ev-seed-001"],
        verification_plan=[
            VerificationStep(
                command=command,
                description="verification step",
                timeout_seconds=timeout,
            )
            for command, timeout in zip(commands, timeouts, strict=True)
        ],
    )


def test_verifier_marks_supported_for_passing_tests(
    git_repo,
    tmp_path: Path,
) -> None:
    before = capture_repository_fingerprint(git_repo.repo)
    with RuntimeVerificationSandbox(
        git_repo.repo,
        work_dir=tmp_path,
    ) as sandbox:
        graph = _graph(
            git_repo,
            [_hypothesis("h-001", "python -m pytest -q tests/test_calc.py")],
        )
        run = RuntimeTestVerifier(sandbox).verify(graph)

    assert len(run.results) == 1
    result = run.results[0]
    assert result.status == VerificationStatus.PASSED
    assert result.outcome == VerificationOutcome.SUPPORTED
    assert result.exit_code == 0
    assert run.graph.hypotheses[0].disposition == HypothesisDisposition.SUPPORTED

    item = run.evidence[0]
    assert item.kind == EvidenceKind.TEST_RESULT
    assert item.agent == AgentRole.RUNTIME_TEST
    assert item.provenance.source == "verification_sandbox"
    assert item.provenance.tool == "runtime_test"
    assert item.provenance.command == "python -m pytest -q tests/test_calc.py"
    assert item.provenance.commit == sandbox.head_sha
    assert result.evidence_ids == [item.id]
    assert item.id in {evidence.id for evidence in run.graph.evidence}

    after = capture_repository_fingerprint(git_repo.repo)
    assert before.model_dump(mode="json") == after.model_dump(mode="json")


def test_verifier_marks_rejected_for_failing_tests(
    git_repo,
    tmp_path: Path,
) -> None:
    with RuntimeVerificationSandbox(
        git_repo.repo,
        work_dir=tmp_path,
    ) as sandbox:
        _write(
            sandbox.work_root,
            "tests/test_fail.py",
            "def test_fail():\n    assert False\n",
        )
        graph = _graph(
            git_repo,
            [_hypothesis("h-002", "python -m pytest -q tests/test_fail.py")],
        )
        run = RuntimeTestVerifier(sandbox).verify(graph)

    result = run.results[0]
    assert result.status == VerificationStatus.FAILED
    assert result.outcome == VerificationOutcome.REJECTED
    assert result.exit_code != 0
    assert run.graph.hypotheses[0].disposition == HypothesisDisposition.REJECTED
    assert "assert False" in result.output_excerpt or result.output_excerpt


def test_verifier_marks_timeout_as_unverified(
    git_repo,
    tmp_path: Path,
) -> None:
    with RuntimeVerificationSandbox(
        git_repo.repo,
        work_dir=tmp_path,
    ) as sandbox:
        _write(
            sandbox.work_root,
            "tests/test_slow.py",
            "import time\n"
            "def test_slow():\n"
            "    time.sleep(30)\n",
        )
        graph = _graph(
            git_repo,
            [
                _hypothesis(
                    "h-003",
                    "python -m pytest -q tests/test_slow.py",
                    timeouts=[1],
                )
            ],
        )
        run = RuntimeTestVerifier(sandbox).verify(graph)

    result = run.results[0]
    assert result.status == VerificationStatus.ERROR
    assert result.outcome == VerificationOutcome.UNVERIFIED
    assert run.graph.hypotheses[0].disposition == HypothesisDisposition.UNVERIFIED


def test_verifier_rejects_general_shell_commands(
    git_repo,
    tmp_path: Path,
) -> None:
    with RuntimeVerificationSandbox(
        git_repo.repo,
        work_dir=tmp_path,
    ) as sandbox:
        graph = _graph(
            git_repo,
            [_hypothesis("h-004", "rm -rf /")],
        )
        run = RuntimeTestVerifier(sandbox).verify(graph)
        assert (sandbox.work_root / "pkg").exists()

    result = run.results[0]
    assert result.status == VerificationStatus.ERROR
    assert result.outcome == VerificationOutcome.UNVERIFIED
    assert run.evidence == []
    assert run.graph.hypotheses[0].disposition == HypothesisDisposition.UNVERIFIED


def test_verifier_leaves_hypothesis_without_plan_unverified(
    git_repo,
    tmp_path: Path,
) -> None:
    with RuntimeVerificationSandbox(
        git_repo.repo,
        work_dir=tmp_path,
    ) as sandbox:
        graph = _graph(git_repo, [_hypothesis("h-005")])
        run = RuntimeTestVerifier(sandbox).verify(graph)
    assert run.results == []
    assert run.graph.hypotheses[0].disposition == HypothesisDisposition.UNVERIFIED


def test_verifier_honors_step_budget(git_repo, tmp_path: Path) -> None:
    with RuntimeVerificationSandbox(
        git_repo.repo,
        work_dir=tmp_path,
    ) as sandbox:
        graph = _graph(
            git_repo,
            [
                _hypothesis("h-006", "python -m pytest -q tests/test_calc.py"),
                _hypothesis("h-007", "python -m pytest -q tests/test_calc.py"),
                _hypothesis("h-008", "python -m pytest -q tests/test_calc.py"),
            ],
        )
        run = RuntimeTestVerifier(sandbox).verify(graph, max_steps=2)
    assert len(run.results) == 2
    dispositions = {
        hypothesis.id: hypothesis.disposition
        for hypothesis in run.graph.hypotheses
    }
    assert dispositions["h-006"] == HypothesisDisposition.SUPPORTED
    assert dispositions["h-007"] == HypothesisDisposition.SUPPORTED
    assert dispositions["h-008"] == HypothesisDisposition.UNVERIFIED
