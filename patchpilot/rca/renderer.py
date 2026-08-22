"""Deterministic Markdown rendering of the final RCA report."""

from __future__ import annotations

from patchpilot.rca.schema import RCAReport


def _location_text(location) -> str:
    text = location.path
    if location.symbol:
        text += f"::{location.symbol}"
    if location.start_line is not None:
        text += f":{location.start_line}"
        if location.end_line is not None:
            text += f"-{location.end_line}"
    return text


def render_rca_markdown(report: RCAReport) -> str:
    """Render the report as bounded, deterministic Markdown."""
    graph = report.evidence_graph
    incident = graph.incident
    lines: list[str] = []
    lines.append(f"# RootTrace RCA Report — {incident.id}")
    lines.append("")
    lines.append(f"- Conclusion: `{report.conclusion.value}`")
    if report.conclusion_summary:
        lines.append(f"- Summary: {report.conclusion_summary}")
    lines.append(f"- Base commit: `{incident.base_commit}`")
    lines.append("")

    lines.append("## Ranked causes")
    lines.append("")
    if report.ranked_causes:
        lines.append("| Rank | Hypothesis | Confidence | Rationale |")
        lines.append("| ---: | --- | --- | --- |")
        for cause in report.ranked_causes:
            rationale = cause.rationale or ""
            lines.append(
                f"| {cause.rank} | `{cause.hypothesis_id}` | "
                f"{cause.confidence.value} | {rationale} |"
            )
    else:
        lines.append("_No ranked cause selected._")
    lines.append("")

    lines.append("## Top-K locations")
    lines.append("")
    if report.top_k_locations:
        lines.append("| # | Location |")
        lines.append("| ---: | --- |")
        for index, location in enumerate(report.top_k_locations, start=1):
            lines.append(f"| {index} | `{_location_text(location)}` |")
    else:
        lines.append("_No locations ranked._")
    lines.append("")

    lines.append("## Causal chain")
    lines.append("")
    if report.causal_chain:
        for link in report.causal_chain:
            hypothesis = (
                f" (`{link.hypothesis_id}`)" if link.hypothesis_id else ""
            )
            lines.append(f"- {link.statement}{hypothesis}")
    else:
        lines.append("_No causal chain reported._")
    lines.append("")

    lines.append("## Verification")
    lines.append("")
    if report.verification:
        lines.append("| Result | Hypothesis | Command | Status | Outcome | Exit |")
        lines.append("| --- | --- | --- | --- | --- | ---: |")
        for result in report.verification:
            command = result.command if len(result.command) <= 80 else (
                result.command[:77] + "..."
            )
            exit_code = "" if result.exit_code is None else str(result.exit_code)
            lines.append(
                f"| `{result.id}` | `{result.hypothesis_id}` | "
                f"`{command}` | {result.status.value} | "
                f"{result.outcome.value} | {exit_code} |"
            )
    else:
        lines.append("_No verification executed._")
    lines.append("")

    if report.suspected_regression is not None:
        lines.append("## Suspected regression")
        lines.append("")
        lines.append(f"- Commit: `{report.suspected_regression.commit}`")
        if report.suspected_regression.summary:
            lines.append(f"- Summary: {report.suspected_regression.summary}")
        lines.append("")

    if report.fix_recommendation is not None:
        lines.append("## Fix recommendation (advisory only)")
        lines.append("")
        if report.fix_recommendation.scope:
            lines.append(f"- Scope: {report.fix_recommendation.scope}")
        for suggestion in report.fix_recommendation.suggestions:
            lines.append(f"- {suggestion}")
        lines.append("")

    lines.append("## Uncertainty")
    lines.append("")
    lines.append(f"- Level: `{report.uncertainty.level.value}`")
    lines.append(
        f"- Insufficient evidence: "
        f"`{'true' if report.uncertainty.insufficient_evidence else 'false'}`"
    )
    for note in report.uncertainty.notes:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Hypotheses")
    lines.append("")
    if graph.hypotheses:
        lines.append("| Id | Disposition | Confidence | Statement |")
        lines.append("| --- | --- | --- | --- |")
        for hypothesis in graph.hypotheses:
            lines.append(
                f"| `{hypothesis.id}` | {hypothesis.disposition.value} | "
                f"{hypothesis.confidence.value} | {hypothesis.statement} |"
            )
    else:
        lines.append("_No hypotheses._")
    lines.append("")

    lines.append("## Evidence")
    lines.append("")
    if graph.evidence:
        lines.append("| Id | Agent | Kind | Observation |")
        lines.append("| --- | --- | --- | --- |")
        for item in graph.evidence:
            observation = item.observation
            if len(observation) > 120:
                observation = observation[:117] + "..."
            lines.append(
                f"| `{item.id}` | {item.agent.value} | "
                f"{item.kind.value} | {observation} |"
            )
    else:
        lines.append("_No evidence collected._")
    lines.append("")

    timing = report.timing
    usage = report.usage
    lines.append("## Timing and usage")
    lines.append("")
    lines.append(
        f"- Timing: total={timing.total_seconds or 'n/a'}s, "
        f"model={timing.model_seconds or 'n/a'}s, "
        f"verification={timing.verification_seconds or 'n/a'}s"
    )
    tokens = (
        f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}"
        if usage.prompt_tokens is not None and usage.completion_tokens is not None
        else "tokens unavailable"
    )
    lines.append(f"- Usage: {usage.llm_calls} calls, {tokens}")
    return "\n".join(lines) + "\n"
