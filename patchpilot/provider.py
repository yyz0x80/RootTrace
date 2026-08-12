"""Provider module for calling OpenAI-compatible APIs.

This module handles communication with OpenAI-compatible API providers,
including request conversion, response parsing, error handling, and retries.
"""

import json
import os
from time import sleep

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError

from patchpilot.models import AssistantTurn, ToolCall

# Load environment variables from .env file if it exists
load_dotenv()


class ToolCallParseError(Exception):
    """Raised when tool call arguments cannot be parsed as JSON."""



class LLMProvider:
    """Provider for OpenAI-compatible API interactions.

    This class handles:
    - API authentication and configuration
    - Message and tool conversion
    - Response parsing and tool call extraction
    - Error handling and retries for rate limits
    """

    def __init__(self) -> None:
        """Initialize the provider with environment configuration.

        Reads:
        - ZHIPU_API_KEY: API authentication key
        - PATCHPILOT_BASE_URL: API base URL
        - PATCHPILOT_MODEL: Model identifier
        """
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY environment variable is not set")

        base_url = os.getenv("PATCHPILOT_BASE_URL")
        if not base_url:
            raise ValueError("PATCHPILOT_BASE_URL environment variable is not set")

        model = os.getenv("PATCHPILOT_MODEL")
        if not model:
            raise ValueError("PATCHPILOT_MODEL environment variable is not set")

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> AssistantTurn:
        """Complete a conversation turn with the LLM.

        Args:
            messages: List of message dictionaries with role and content
            tools: List of tool schemas available for the model to call

        Returns:
            AssistantTurn containing the response content and any tool calls

        Raises:
            ToolCallParseError: If tool call arguments cannot be parsed as JSON
            OpenAIError: For other API-related errors after retries are exhausted
        """
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                )

                message = response.choices[0].message
                content = message.content
                tool_calls = []

                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            raise ToolCallParseError(
                                f"Failed to parse tool call arguments as JSON: "
                                f"tool={tool_call.function.name}, "
                                f"arguments={tool_call.function.arguments}"
                            ) from e

                        tool_calls.append(
                            ToolCall(
                                id=tool_call.id,
                                name=tool_call.function.name,
                                arguments=arguments,
                            )
                        )

                return AssistantTurn(content=content, tool_calls=tool_calls)

            except RateLimitError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    sleep(2**attempt)
                    continue
                else:
                    raise OpenAIError(
                        f"Rate limit exceeded after {max_retries} retry attempts"
                    ) from e

            except OpenAIError:
                # Non-rate-limit errors are not retried
                raise

        # This should not be reached, but kept for type safety
        raise OpenAIError("Unexpected error in completion logic") from last_error

    def generate_text(self, prompt: str) -> str:
        """Generate text response from a simple prompt.

        This is a convenience method for simple text generation without tools.
        It adapts the simple prompt-response interface to the structured
        complete() method.

        Args:
            prompt: The input prompt text.

        Returns:
            The generated text response.

        Raises:
            ValueError: If the response content is None.
            OpenAIError: For API-related errors.
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.complete(messages=messages, tools=[])
        
        if response.content is None:
            raise ValueError("LLM returned None content")
        
        return response.content
