#!/usr/bin/env python3
"""Simple test to verify provider connectivity.

This script tests:
1. Provider initialization
2. Simple text generation
3. Tool call functionality
4. Token usage tracking
"""

import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from patchpilot.provider import LLMProvider, create_provider_from_config


def test_provider():
    """Test provider with API."""

    # Check for API key
    api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: ZHIPU_API_KEY or OPENAI_API_KEY environment variable not set")
        print("Please set it with: export ZHIPU_API_KEY='your-key-here'")
        return False

    print("Found API key")

    try:
        # Test 1: Provider initialization
        print("\nTest 1: Provider initialization")
        provider = create_provider_from_config()
        print("Provider created successfully")
        print(f"   Model: {provider.model}")
        
        # Test 2: Simple text generation
        print("\nTest 2: Simple text generation")
        test_prompt = "What is 2+2? Answer with just the number."
        response = provider.generate_text(test_prompt)
        print("Text generation successful")
        print(f"   Prompt: {test_prompt}")
        print(f"   Response: {response}")
        print(f"   LLM calls: {provider.llm_call_count}")
        print(f"   Prompt tokens: {provider.prompt_tokens}")
        print(f"   Completion tokens: {provider.completion_tokens}")
        
        # Test 3: Tool call simulation
        print("\nTest 3: Tool call functionality")
        messages = [
            {"role": "user", "content": "What's the weather like today?"}
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        response = provider.complete(messages=messages, tools=tools)
        print("Tool call handling successful")
        print(f"   Response content: {response.content}")
        print(f"   Tool calls: {len(response.tool_calls)}")
        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"   - Tool: {tool_call.name}, Args: {tool_call.arguments}")
        
        # Test 4: Token tracking
        print("\nTest 4: Token usage tracking")
        print(f"   Total LLM calls: {provider.llm_call_count}")
        print(f"   Total prompt tokens: {provider.prompt_tokens}")
        print(f"   Total completion tokens: {provider.completion_tokens}")
        if provider.prompt_tokens and provider.completion_tokens:
            total = provider.prompt_tokens + provider.completion_tokens
            print(f"   Total tokens: {total}")

        print("\nAll tests passed! Provider is working correctly.")
        return True

    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        print("Please install required packages: pip install openai python-dotenv")
        return False
    except Exception as e:
        print(f"ERROR: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing Provider Connectivity")
    print("=" * 50)

    success = test_provider()

    print("\n" + "=" * 50)
    if success:
        print("Provider test completed successfully")
        sys.exit(0)
    else:
        print("Provider test failed")
        sys.exit(1)
