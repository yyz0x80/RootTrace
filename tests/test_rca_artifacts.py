"""Focused tests for RootTrace RCA artifact persistence."""

import json

import pytest

from patchpilot.rca.artifacts import (
    ARTIFACT_EVIDENCE_GRAPH,
    ARTIFACT_INCIDENT,
    ArtifactError,
    ArtifactWriter,
    model_to_json,
    validate_artifact_name,
    write_artifact,
)
from patchpilot.rca.schema import (
    EvidenceGraph,
    IncidentInput,
    Provenance,
    RCAReport,
    ReportConclusion,
)


def make_incident(**overrides) -> IncidentInput:
    fields = {
        "id": "inc-1",
        "repo": "demo/repo",
        "base_commit": "a" * 40,
        "problem": "Crash when loading the module",
        "provenance": Provenance(source="issue.md:1"),
    }
    fields.update(overrides)
    return IncidentInput(**fields)


def make_graph(**overrides) -> EvidenceGraph:
    fields = {
        "incident": make_incident(),
        "findings": [],
        "evidence": [],
        "hypotheses": [],
        "edges": [],
    }
    fields.update(overrides)
    return EvidenceGraph(**fields)


def make_report() -> RCAReport:
    graph = make_graph()
    return RCAReport(
        id="report-1",
        incident_id=graph.incident.id,
        evidence_graph=graph,
        conclusion=ReportConclusion.INSUFFICIENT_EVIDENCE,
    )


def test_write_model_creates_round_trippable_artifact(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path)
    graph = make_graph()
    path = writer.write_model(ARTIFACT_EVIDENCE_GRAPH, graph)
    assert path == tmp_path / ARTIFACT_EVIDENCE_GRAPH
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert EvidenceGraph.model_validate(data) == graph


def test_write_nested_artifact(tmp_path) -> None:
    path = ArtifactWriter(tmp_path).write_model(ARTIFACT_INCIDENT, make_incident())
    assert path == tmp_path / ARTIFACT_INCIDENT
    assert path.exists()


def test_writes_stay_inside_output_dir(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path)
    for bad in (
        "../escape.json",
        "a/../../escape.json",
        "/tmp/escape.json",
        "..",
        "a\\b.json",
        "agents/",
    ):
        with pytest.raises(ArtifactError):
            writer.write_model(bad, make_incident())
    assert not (tmp_path.parent / "escape.json").exists()


def test_symlink_escape_rejected(tmp_path) -> None:
    outside = tmp_path.parent / "roottrace-outside"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    writer = ArtifactWriter(tmp_path)
    with pytest.raises(ArtifactError):
        writer.write_model("link/x.json", make_incident())
    assert not (outside / "x.json").exists()


def test_write_is_deterministic(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path)
    first = writer.write_model(ARTIFACT_INCIDENT, make_incident())
    second = writer.write_model(ARTIFACT_INCIDENT, make_incident())
    assert first.read_bytes() == second.read_bytes()


def test_output_dir_is_created(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path / "nested" / "out")
    path = writer.write_model(ARTIFACT_INCIDENT, make_incident())
    assert path.exists()


def test_model_to_json_is_deterministic() -> None:
    report = make_report()
    assert model_to_json(report) == model_to_json(report.model_copy(deep=True))


def test_write_artifact_helper(tmp_path) -> None:
    path = write_artifact(tmp_path, ARTIFACT_INCIDENT, make_incident())
    assert path.exists()


def test_validate_artifact_name_accepts_nested_names() -> None:
    assert validate_artifact_name("agents/issue_ci.json") == "agents/issue_ci.json"
