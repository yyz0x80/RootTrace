"""Deterministic localization baseline without LLM agents (M9-C).

The baseline replays the deterministic context builder and uses the ranked
source snippets as predicted files. It makes zero model calls, which makes it
the reference point for every ablation comparison.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from evaluation.rca.runner import RootTraceOutcome
from patchpilot.rca.incident_loader import LoadedIncident
from patchpilot.rca.schema import IncidentInput

MAX_PREDICTED_FILES = 10
REPORT_FILE = "rca_report.json"


class DeterministicBaselineClient:
    """RootTrace client that predicts files from deterministic context only."""

    def run(
        self,
        *,
        case_id: str,
        repo: Path,
        incident: IncidentInput,
        output_dir: Path,
        model: str | None,
    ) -> RootTraceOutcome:
        del case_id, model
        from patchpilot.rca.context_builder import build_incident_context

        loaded = LoadedIncident(incident=incident)
        started = time.monotonic()
        context = build_incident_context(loaded, repo)
        files: list[str] = []
        for snippet in context.snippets:
            if snippet.path not in files:
                files.append(snippet.path)
            if len(files) >= MAX_PREDICTED_FILES:
                break
        report = {
            "top_k_locations": [{"path": path} for path in files],
            "baseline": {
                "mode": "deterministic_context_ranking",
                "candidate_snippets": len(context.snippets),
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / REPORT_FILE).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        elapsed = time.monotonic() - started
        return RootTraceOutcome(
            status="completed",
            report=report,
            latency_seconds=elapsed,
            llm_calls=0,
            prompt_tokens=None,
            completion_tokens=None,
        )
