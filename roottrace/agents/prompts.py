"""Bounded role prompts for the RootTrace RCA agents.

Prompt builders serialize only the deterministic context prepared by the
context builder; every prompt is bounded and carries explicit truncation
metadata. The Lead plans an investigation without committing to a root cause;
the three evidence Specialists stay inside their own evidence domain and cite
evidence IDs.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from roottrace.agents.schema import PlanQuestion
from roottrace.evidence.graph import EvidenceGraph
from roottrace.incident.context import IncidentContext
from roottrace.verification.schema import VerificationResult

MAX_PROMPT_CHARS = 120_000
_TRUNCATION_MARKER = "\n... [prompt truncated]"


class RetrievalHintsLike(Protocol):
    """Duck-typed retrieval hints: bounded prior-case evidence, hints only."""

    mode: str
    results: list[Any]


def _bounded_json(data: object, limit: int = MAX_PROMPT_CHARS) -> str:
    """Serialize deterministically and bound the resulting prompt text."""
    text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATION_MARKER


def _question_lines(questions: list[PlanQuestion]) -> list[dict[str, str]]:
    return [{"id": question.id, "text": question.text} for question in questions]


LEAD_SYSTEM_PROMPT = """\
You are the RootTrace Lead Agent for a root cause analysis (RCA) investigation.
Your job is to plan the investigation, not to diagnose it.

Rules:
- Produce bounded investigation questions and assign them to the three
  evidence Specialists (issue_ci, code, git_history).
- Do NOT state or imply a root cause, a likely fix, or a suspected regression
  before evidence has been gathered.
- Prefer deterministic evidence collection over speculation.
- Selected pull request review threads are bounded advisory evidence. Treat
  reviewer statements as claims to investigate, not as established facts.

Output only one JSON object with this schema:
{
  "questions": [
    {
      "id": "<unique-question-id>",
      "text": "concise question",
      "assigned_agents": ["issue_ci"]
    }
  ]
}

Each question must be assigned to at least one of: "issue_ci", "code",
"git_history". At most 10 questions.
"""


ISSUE_CI_SYSTEM_PROMPT = """\
You are the RootTrace Issue/CI Specialist.
Your evidence domain is the incident itself: issue text, stack traces, CI log
excerpts, and failure signatures.

Rules:
- You may only call the read_external_log tool. Never call code-search, file
  read, symbol, git, or runtime tools.
- You do not produce a root cause. Report symptoms and failure signatures with
  explicit uncertainty.
- Review-thread comments are separate, bounded evidence. Keep each comment's
  author, thread, location, and provenance distinct, and distinguish replies
  that support or contradict the root comment.
- Every factual claim must cite one of the provided evidence ids.
- When external logs are provided for this run, read_external_log accepts the
  paths "ci.log" and "stack_trace.log".

Tool results identify their evidence id as "Evidence id: ev-issue_ci-<n>".
Your final answer is one JSON object:
{
  "status": "completed|partial|failed",
  "ranked_locations": [{"path": "repo/relative/path", "symbol": "name"}],
  "evidence_ids": ["<existing-evidence-id>"],
  "uncertainty": "low|medium|high",
  "uncertainty_note": "short note"
}
ranked_locations is optional for this role; evidence_ids must reference only
ids you actually saw.
"""


CODE_SYSTEM_PROMPT = """\
You are the RootTrace Code Specialist.
Your evidence domain is repository code: likely files, functions, symbols, and
code paths.

Rules:
- You may only call search_code, read_file, and inspect_symbols. Never call
  git, external-log, or runtime tools.
- You do not produce a root cause. Localize suspicious code with explicit
  uncertainty.
- Review-comment anchors are bounded leads only. Verify every reviewer claim
  against the checked-out code with the available repository tools; do not
  treat a comment or line number as proof of a defect.
- Every factual claim must cite one of the provided evidence ids.

Tool results identify their evidence id as "Evidence id: ev-code-<n>".
Your final answer is one JSON object:
{
  "status": "completed|partial|failed",
  "ranked_locations": [
    {
      "path": "<repo-relative-path>",
      "symbol": "<symbol-name>",
      "start_line": 8,
      "end_line": 9
    }
  ],
  "evidence_ids": ["<existing-evidence-id>"],
  "uncertainty": "low|medium|high",
  "uncertainty_note": "short note"
}
ranked_locations are repo-relative paths. evidence_ids must reference only ids
you actually saw.
"""


GIT_HISTORY_SYSTEM_PROMPT = """\
You are the RootTrace Git History Specialist.
Your evidence domain is repository history: relevant changes, blame lines, and
suspected regression commits.

Rules:
- You may only call git_history, git_blame, and git_show. Never call
  code-search, file read, symbol, external-log, or runtime tools.
- You do not produce a root cause. Report historical evidence with explicit
  uncertainty; a regression is suspected only when history supports it.
- Follow the deterministic Git verification policy in the context. When it is
  disabled, inspect only the base commit and do not expand history.
- When it is enabled, prioritize the listed candidate commits and paths, stay
  within the bounded history, and never treat a PR head commit as a regression
  commit because the checkout is pinned to the base revision.
- The prepared Git search opens history progressively and includes only its
  final Top-K candidates. Treat 50 commits as a hard ceiling, not a default
  amount to inspect. Do not repeat a broad unscoped history scan.
- A path-only or commit-message-only candidate is a weak lead, not regression
  evidence. Inspect it with git_show or a scoped history/blame call before
  citing it as a suspected regression.
- Review comments are not Git evidence. Use their path, commit, and mapping
  metadata only to scope history inspection; do not consume their discussion
  bodies as history evidence.
- Every factual claim must cite one of the provided evidence ids.

Tool results identify their evidence id as "Evidence id: ev-git_history-<n>".
Your final answer is one JSON object:
{
  "status": "completed|partial|failed",
  "ranked_locations": [{"path": "<repo-relative-path>"}],
  "evidence_ids": ["<existing-evidence-id>"],
  "uncertainty": "low|medium|high",
  "uncertainty_note": "short note"
}
ranked_locations are repo-relative paths. evidence_ids must reference only ids
you actually saw.
"""


HYPOTHESES_SYSTEM_PROMPT = """\
You are the RootTrace Lead Agent generating falsifiable root-cause hypotheses
after evidence has been gathered.

Rules:
- Hypotheses must be falsifiable and grounded in the provided evidence graph.
- Every hypothesis must cite supporting evidence ids that exist in the graph;
  contradicting evidence ids are optional.
- Rank hypotheses from most to least likely.
- Each hypothesis needs a bounded verification plan using only sandbox test
  commands of the form "python -m pytest <relative test target> [flags]".
- Set expect_failure to true when the verification command should exit
  non-zero to confirm the hypothesis (e.g., reproducing the reported failure
  on the analyzed commit). Set it to false when a passing command confirms the
  hypothesis.
- Do not propose code edits, patches, or repository mutations.
- Confidence only ranks evidence; it never replaces it.

Output only one JSON object with this schema:
{
  "hypotheses": [
    {
      "statement": "falsifiable root-cause claim",
      "locations": [{"path": "<repo-relative-path>", "symbol": "<symbol-name>"}],
      "supporting_evidence_ids": ["<existing-evidence-id>"],
      "contradicting_evidence_ids": [],
      "verification_plan": [
        {
          "command": "python -m pytest -q <relative-test-target>",
          "description": "short description",
          "timeout_seconds": 60,
          "expect_failure": false
        }
      ],
      "confidence": "low|medium|high"
    }
  ]
}

Return at most 5 hypotheses. Evidence ids must be chosen only from the ids
listed in the EVIDENCE GRAPH section.
"""


def build_hypotheses_prompt(
    graph: EvidenceGraph,
    *,
    retrieval_hints: RetrievalHintsLike | None = None,
) -> str:
    """Build the bounded hypothesis-generation prompt from the evidence graph.

    ``retrieval_hints`` are optional prior-case hints from shared memory; they
    are presented as bounded context and never override current-repository
    evidence.
    """
    incident = graph.incident
    evidence = [
        {
            "id": item.id,
            "agent": item.agent.value,
            "kind": item.kind.value,
            "commit_ids": item.commit_ids,
            "observation": item.observation,
            "provenance": item.provenance.model_dump(mode="json"),
            "location": item.location.model_dump(mode="json")
            if item.location is not None
            else None,
            "excerpt": item.excerpt[:600]
            + ("..." if len(item.excerpt) > 600 else ""),
        }
        for item in graph.evidence
    ]
    findings = [
        {
            "agent": finding.agent.value,
            "status": finding.status.value,
            "ranked_locations": [
                location.model_dump(mode="json")
                for location in finding.ranked_locations
            ],
            "evidence_ids": finding.evidence_ids,
            "uncertainty": finding.uncertainty.value,
            "uncertainty_note": finding.uncertainty_note,
        }
        for finding in graph.findings
    ]
    data = {
        "incident": {
            "id": incident.id,
            "title": incident.title,
            "problem": incident.problem,
            "base_commit": incident.base_commit,
            "resource_kind": incident.resource_kind,
            "git_verification_policy": incident.git_verification_policy.model_dump(
                mode="json"
            ),
        },
        "findings": findings,
        "evidence": evidence,
    }
    prompt = "\n\nEVIDENCE GRAPH:\n" + _bounded_json(data)
    if retrieval_hints is not None and retrieval_hints.results:
        hints_payload = [
            {
                "id": result.id,
                "similarity": result.similarity,
                "locations": result.locations[:10],
                "summary": result.summary,
                "source": result.source,
            }
            for result in retrieval_hints.results[:5]
        ]
        prompt += (
            "\n\nRETRIEVED HISTORICAL HINTS "
            "(prior cases only; never override current evidence):\n"
            + _bounded_json({"mode": retrieval_hints.mode, "results": hints_payload})
        )
    return prompt


FINAL_REPORT_SYSTEM_PROMPT = """\
You are the RootTrace Lead Agent producing the final RCA report after evidence
gathering and runtime verification.

Rules:
- Choose the best-supported cause, or report insufficient evidence.
- Rank causes only when the evidence and verification support them; cite
  evidence ids that exist in the provided graph.
- Never invent evidence ids or hypothesis ids; every cited id must appear
  verbatim in the INVESTIGATION STATE.
- evidence_ids fields (ranked_causes, causal_chain, suspected_regression) may
  reference only ids from the evidence list in INVESTIGATION STATE; never cite
  verification result ids (ver-*) and never invent ids.
- Every ranked cause must reference a hypothesis with at least one verification
  result whose outcome is ``supported`` and cite at least one evidence id from
  that supported result.
- Never rank a hypothesis whose verification outcome is rejected. When no
  hypothesis is supported by verification, conclude insufficient_evidence.
- Supported runtime verification constrains only a confirmed root cause,
  ranked_causes, and a ``root_cause_identified`` conclusion. It does not make
  localization unavailable when the conclusion is ``insufficient_evidence``.
- Even with ``conclusion=insufficient_evidence`` and an empty ranked_causes
  list, populate top_k_locations with evidence-grounded candidates when
  possible. Candidates may come from non-rejected hypothesis locations or
  specialist ranked_locations; never use locations from rejected hypotheses.
- Report a suspected regression commit only when the INVESTIGATION STATE
  contains real Git evidence: an item from the git_history agent whose tool
  and evidence kind match git_history/GIT_LOG, git_show/GIT_DIFF, or
  git_blame/GIT_BLAME, and whose commit_ids contains a real commit SHA.
- git_verification_policy.enabled controls only the bounded Git search budget
  and prioritization; it never determines whether a regression is admissible.
  Admissibility depends solely on qualifying Git evidence.
- suspected_regression.commit must be an exact 7-64 character hexadecimal SHA
  found in the cited evidence. When no qualifying Git evidence is available,
  set suspected_regression to null or omit it entirely. Never emit an empty
  object, empty, or placeholder commit.
- Fix recommendations are advisory text and scope only. Never embed diffs,
  patches, edit commands, or repository mutation commands.
- Preserve uncertainty from partial worker failures and unverified
  hypotheses.

Output only one JSON object with this schema:
{
  "conclusion": "root_cause_identified|insufficient_evidence",
  "conclusion_summary": "short evidence-backed summary",
  "ranked_causes": [
    {
      "rank": 1,
      "hypothesis_id": "<existing-hypothesis-id>",
      "confidence": "low|medium|high",
      "rationale": "short rationale",
      "evidence_ids": ["<existing-evidence-id>"]
    }
  ],
  "top_k_locations": [
    {
      "path": "<repo-relative-path>",
      "symbol": "<symbol-name>",
      "start_line": 8
    }
  ],
  "causal_chain": [
    {
      "statement": "causal step",
      "hypothesis_id": "<existing-hypothesis-id>",
      "evidence_ids": ["<existing-evidence-id>"]
    }
  ],
  "suspected_regression": null,
  "fix_recommendation": {
    "scope": "advisory fix scope",
    "suggestions": ["advisory suggestion text"],
    "locations": [{"path": "<repo-relative-path>", "symbol": "<symbol-name>"}]
  },
  "uncertainty": {
    "level": "low|medium|high",
    "insufficient_evidence": false,
    "notes": ["short note"]
  }
}

Rules for fields:
- conclusion root_cause_identified requires at least one ranked cause.
- All hypothesis_id and evidence_id values must exist in the provided context.
- suspected_regression is optional and only when supported by qualifying Git
  evidence and a real commit SHA. Without that evidence, use null or omit it;
  do not use {}.
- fix_recommendation is optional and strictly advisory.
"""


def build_final_report_prompt(
    graph: EvidenceGraph,
    results: list[VerificationResult],
) -> str:
    """Build the bounded final-synthesis prompt from graph and verification."""
    incident = graph.incident
    hypotheses = [
        {
            "id": hypothesis.id,
            "statement": hypothesis.statement,
            "disposition": hypothesis.disposition.value,
            "locations": [
                location.model_dump(mode="json")
                for location in hypothesis.locations
            ],
            "supporting_evidence_ids": hypothesis.supporting_evidence_ids,
            "contradicting_evidence_ids": hypothesis.contradicting_evidence_ids,
            "confidence": hypothesis.confidence.value,
        }
        for hypothesis in graph.hypotheses
    ]
    evidence = [
        {
            "id": item.id,
            "agent": item.agent.value,
            "kind": item.kind.value,
            "commit_ids": item.commit_ids,
            "observation": item.observation,
            "provenance": item.provenance.model_dump(mode="json"),
            "location": item.location.model_dump(mode="json")
            if item.location is not None
            else None,
            "excerpt": item.excerpt[:600]
            + ("..." if len(item.excerpt) > 600 else ""),
        }
        for item in graph.evidence
    ]
    verification = [
        {
            "id": result.id,
            "hypothesis_id": result.hypothesis_id,
            "command": result.command,
            "status": result.status.value,
            "outcome": result.outcome.value,
            "exit_code": result.exit_code,
            "evidence_ids": result.evidence_ids,
        }
        for result in results
    ]
    data = {
        "incident": {
            "id": incident.id,
            "title": incident.title,
            "problem": incident.problem,
            "base_commit": incident.base_commit,
            "resource_kind": incident.resource_kind,
            "git_verification_policy": incident.git_verification_policy.model_dump(
                mode="json"
            ),
        },
        "hypotheses": hypotheses,
        "verification": verification,
        "evidence": evidence,
    }
    evidence_agents = sorted({item.agent.value for item in graph.evidence})
    domain_rule = (
        "\n\nEVIDENCE DOMAIN RULE:\n"
        f"Evidence was gathered only by agents: {evidence_agents}.\n"
        "Cite only evidence ids listed under evidence above; never invent ids.\n"
        "Include suspected_regression only for real Git evidence with a matching "
        "Git tool/kind pair and at least one commit SHA; otherwise use null or "
        "omit it, never {}.\n"
        "If 'issue_ci' evidence is absent, do not cite issue/CI evidence ids."
    )
    return "\n\nINVESTIGATION STATE:\n" + _bounded_json(data) + domain_rule


def _incident_view(context: IncidentContext) -> dict[str, object]:
    incident = context.incident
    return {
        "id": incident.id,
        "title": incident.title,
        "problem": incident.problem,
        "repo": incident.repo,
        "base_commit": incident.base_commit,
        "resource_kind": incident.resource_kind,
        "labels": incident.labels,
        "changed_files": incident.changed_files,
        "git_verification_policy": incident.git_verification_policy.model_dump(
            mode="json"
        ),
        "logs": list(incident.logs),
        "review_threads": [
            thread.model_dump(mode="json") for thread in incident.review_threads
        ],
        "review_comment_truncation": incident.review_comment_truncation.model_dump(
            mode="json"
        ),
        "signals": context.signals.model_dump(mode="json"),
        "truncation": context.truncation.model_dump(mode="json"),
    }


def _review_anchor_view(
    context: IncidentContext,
    *,
    include_excerpt: bool,
    excerpt_chars: int = 300,
) -> list[dict[str, object]]:
    """Return bounded review anchors for specialists that need less context."""
    anchors: list[dict[str, object]] = []
    for thread in context.incident.review_threads:
        for comment in thread.comments:
            location = comment.location.model_dump(mode="json") if comment.location else None
            anchor: dict[str, object] = {
                "evidence_id": comment.id,
                "comment_id": comment.comment_id,
                "thread_id": comment.thread_id,
                "location": location,
                "location_mapping": comment.location_mapping,
                "commit": comment.provenance.commit,
                "line": comment.line,
                "start_line": comment.start_line,
                "original_line": comment.original_line,
                "original_start_line": comment.original_start_line,
            }
            if include_excerpt:
                anchor["excerpt"] = comment.excerpt[:excerpt_chars]
            anchors.append(anchor)
    return anchors


def _code_view(context: IncidentContext) -> dict[str, object]:
    repository = context.repository
    return {
        "base_commit": context.incident.base_commit,
        "tracked_files": repository.tracked_files,
        "python_files": repository.python_files,
        "test_files": repository.test_files,
        "config_files": repository.config_files,
        "python_file_list": repository.python_file_list,
        "test_file_list": repository.test_file_list,
        "config_file_list": repository.config_file_list,
        "signals": context.signals.model_dump(mode="json"),
        "snippets": [snippet.model_dump(mode="json") for snippet in context.snippets],
        "diff_excerpt": context.diff_excerpt,
        "review_comment_anchors": _review_anchor_view(
            context,
            include_excerpt=True,
        ),
    }


def build_lead_prompt(context: IncidentContext) -> str:
    """Build the bounded Lead planning prompt from deterministic context."""
    data = {
        "incident": _incident_view(context),
        "repository": {
            "tracked_files": context.repository.tracked_files,
            "python_files": context.repository.python_files,
            "test_files": context.repository.test_files,
            "config_files": context.repository.config_files,
        },
        "snippets": [snippet.model_dump(mode="json") for snippet in context.snippets],
        "diff_excerpt": context.diff_excerpt,
    }
    return "\n\nDETERMINISTIC CONTEXT:\n" + _bounded_json(data)


def build_issue_ci_prompt(
    context: IncidentContext,
    questions: list[PlanQuestion],
) -> str:
    """Build the bounded Issue/CI specialist prompt."""
    data = {
        "incident": _incident_view(context),
        "questions": _question_lines(questions),
    }
    return "\n\nINCIDENT CONTEXT:\n" + _bounded_json(data)


def build_code_prompt(
    context: IncidentContext,
    questions: list[PlanQuestion],
) -> str:
    """Build the bounded Code specialist prompt."""
    data = {
        "code_context": _code_view(context),
        "questions": _question_lines(questions),
    }
    return "\n\nCODE CONTEXT:\n" + _bounded_json(data)


def build_git_history_prompt(
    context: IncidentContext,
    questions: list[PlanQuestion],
) -> str:
    """Build the bounded Git History specialist prompt."""
    data = {
        "base_commit": context.incident.base_commit,
        "resource_kind": context.incident.resource_kind,
        "git_verification_policy": context.incident.git_verification_policy.model_dump(
            mode="json"
        ),
        "signals": context.signals.model_dump(mode="json"),
        "tracked_files": context.repository.tracked_files,
        "python_file_list": context.repository.python_file_list,
        "snippet_paths": [
            {"path": snippet.path, "rank": snippet.rank}
            for snippet in context.snippets
        ],
        "diff_excerpt": context.diff_excerpt,
        "candidate_paths": context.incident.git_verification_policy.candidate_paths,
        "candidate_commits": context.incident.git_verification_policy.candidate_commits,
        "review_comment_anchors": _review_anchor_view(
            context,
            include_excerpt=False,
        ),
        "max_tool_calls": context.incident.git_verification_policy.max_tool_calls,
        "questions": _question_lines(questions),
    }
    return "\n\nHISTORY CONTEXT:\n" + _bounded_json(data)
