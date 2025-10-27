#!/usr/bin/env python
"""Test script to verify BaseAgent methods."""

from agent import BaseAgent


def test_extract_tool_calls():
    """Test extract_tool_calls method."""
    agent = BaseAgent()

    # Test XML format (Qwen style)
    message1 = """<tool_call>
{"name": "get_current_time", "arguments": {}}
</tool_call>"""

    result1 = agent.extract_tool_calls(message1)
    print("Test 1 - XML format:")
    print(f"  Input: {message1}")
    print(f"  Result: {result1}")
    print()

    # Test JSON array format
    message2 = '[{"name": "test_func", "arguments": {"param": "value"}}]'
    result2 = agent.extract_tool_calls(message2)
    print("Test 2 - JSON array format:")
    print(f"  Input: {message2}")
    print(f"  Result: {result2}")
    print()

    # Test message with no tool calls
    message3 = "The current time is 14:30"
    result3 = agent.extract_tool_calls(message3)
    print("Test 3 - No tool calls:")
    print(f"  Input: {message3}")
    print(f"  Result: {result3}")
    print()


def test_execute_tool_call():
    """Test execute_tool_call method."""

    # Create a simple test function
    def test_add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Create agent with the test function
    agent = BaseAgent(tools=[test_add])

    # Test valid call
    call1 = {"name": "test_add", "arguments": {"a": 5, "b": 3}}
    result1 = agent.execute_tool_call(call1)
    print("Test 4 - Valid tool call:")
    print(f"  Input: {call1}")
    print(f"  Result: {result1}")
    print()

    # Test unknown tool
    call2 = {"name": "unknown_func", "arguments": {}}
    result2 = agent.execute_tool_call(call2)
    print("Test 5 - Unknown tool:")
    print(f"  Input: {call2}")
    print(f"  Result: {result2}")
    print()


def test_is_ollama_available():
    """Test is_ollama_available method."""
    agent = BaseAgent(model_name="qwen2.5:1.5b")

    available = agent.is_ollama_available()
    print(f"Test 6 - Ollama availability:")
    print(f"  Model: qwen2.5:1.5b")
    print(f"  Available: {available}")
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing BaseAgent Methods")
    print("=" * 80)
    print()

    test_extract_tool_calls()
    test_execute_tool_call()
    test_is_ollama_available()

    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)
