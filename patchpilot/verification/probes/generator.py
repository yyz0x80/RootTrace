"""Generator for model-generated acceptance probes.

This module handles the generation of acceptance probes using LLMs.
Probes are structured verification scripts that test specific aspects
of code changes.
"""

from __future__ import annotations

import json
from typing import Any

from patchpilot.provider import LLMProvider
from patchpilot.verification.probes.schema import (
    AcceptanceProbe,
    ProbeStep,
    ProbeType,
)


class ProbeGenerator:
    """Generate acceptance probes using LLM.

    The generator uses the configured LLMProvider to generate structured
    acceptance probes based on issue descriptions and acceptance criteria.
    """

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize the probe generator.

        Args:
            provider: Configured LLMProvider instance for LLM calls
        """
        self.provider = provider

    def generate_probe(
        self,
        issue_description: str,
        acceptance_criteria: list[str],
        target_function: str,
        probe_type: ProbeType,
    ) -> AcceptanceProbe:
        """Generate a single acceptance probe.

        Args:
            issue_description: Description of the issue being addressed
            acceptance_criteria: List of acceptance criteria to validate
            target_function: Target function or method to test
            probe_type: Type of probe to generate

        Returns:
            AcceptanceProbe instance with generated verification steps
        """
        prompt = self._build_generation_prompt(
            issue_description,
            acceptance_criteria,
            target_function,
            probe_type,
        )

        # Create a simple JSON schema in the prompt
        json_instruction = f"""
Respond with a JSON object matching this schema:
{json.dumps(self._get_probe_schema(), indent=2)}

Your response should be ONLY the JSON object, no other text.
"""

        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt(),
            },
            {
                "role": "user",
                "content": prompt + "\n\n" + json_instruction,
            },
        ]

        response = self.provider.complete(messages=messages, tools=[])

        if response.content is None:
            raise ValueError("LLM returned None content for probe generation")

        return self._parse_probe_response(response.content, target_function, probe_type)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for probe generation.

        Returns:
            System prompt instructions for the LLM
        """
        return """You are an expert at generating verification probes for code changes.
Your task is to create structured acceptance probes that test specific aspects
of code changes without becoming part of the patch itself.

Generated probes must:
- Use only safe, whitelisted Python operations
- Not modify files or system state
- Be executable in isolated temporary directories
- Provide clear pass/fail results
- Include descriptive error messages on failure"""

    def _build_generation_prompt(
        self,
        issue_description: str,
        acceptance_criteria: list[str],
        target_function: str,
        probe_type: ProbeType,
    ) -> str:
        """Build the user prompt for probe generation.

        Args:
            issue_description: Description of the issue
            acceptance_criteria: List of acceptance criteria
            target_function: Target function to test
            probe_type: Type of probe to generate

        Returns:
            Formatted prompt for the LLM
        """
        criteria_text = "\n".join(
            f"- {criterion}" for criterion in acceptance_criteria
        )

        return f"""Generate an acceptance probe for the following code change:

Issue Description:
{issue_description}

Acceptance Criteria:
{criteria_text}

Target Function: {target_function}
Probe Type: {probe_type.value}

Generate a probe with 3-5 verification steps that test the specified aspect.
Each step should be self-contained and produce a clear pass/fail result."""

    def _get_probe_schema(self) -> dict[str, Any]:
        """Get the JSON schema for probe generation.

        Returns:
            JSON schema for structured probe output
        """
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "code": {"type": "string"},
                            "expected_outcome": {"type": "string"},
                            "tolerance": {"type": "number"},
                        },
                        "required": ["description", "code", "expected_outcome"],
                    },
                },
                "setup_code": {"type": "string"},
                "teardown_code": {"type": "string"},
            },
            "required": ["name", "description", "steps"],
        }

    def _parse_probe_response(
        self,
        response: str,
        target_function: str,
        probe_type: ProbeType,
    ) -> AcceptanceProbe:
        """Parse the LLM response into an AcceptanceProbe.

        Args:
            response: Raw response from the LLM
            target_function: Target function being tested
            probe_type: Type of probe generated

        Returns:
            AcceptanceProbe instance
        """
        import uuid

        data = json.loads(response)

        steps = [
            ProbeStep(
                description=step["description"],
                code=step["code"],
                expected_outcome=step["expected_outcome"],
                tolerance=step.get("tolerance"),
            )
            for step in data.get("steps", [])
        ]

        return AcceptanceProbe(
            id=str(uuid.uuid4()),
            name=data["name"],
            description=data["description"],
            probe_type=probe_type,
            target_function=target_function,
            steps=steps,
            setup_code=data.get("setup_code", ""),
            teardown_code=data.get("teardown_code", ""),
        )
