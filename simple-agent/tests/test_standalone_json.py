#!/usr/bin/env python
"""Test standalone JSON tool call extraction."""

from agent import BaseAgent


def test_standalone_json_extraction():
    """Test that standalone JSON objects are recognized as tool calls."""
    agent = BaseAgent()

    # Test 1: Standalone JSON with arguments
    message1 = '{"name": "get_current_time", "arguments": {}}'
    result1 = agent.extract_tool_calls(message1)
    print("Test 1 - Standalone JSON with empty arguments:")
    print(f"  Input: {message1}")
    print(f"  Result: {result1}")
    print(f"  Success: {len(result1) == 1 and result1[0]['name'] == 'get_current_time'}")
    print()

    # Test 2: Standalone JSON with actual arguments
    message2 = '{"name": "test_func", "arguments": {"param1": "value1", "param2": 42}}'
    result2 = agent.extract_tool_calls(message2)
    print("Test 2 - Standalone JSON with arguments:")
    print(f"  Input: {message2}")
    print(f"  Result: {result2}")
    print(f"  Success: {len(result2) == 1 and result2[0]['arguments']['param1'] == 'value1'}")
    print()

    # Test 3: Standalone JSON without arguments field
    message3 = '{"name": "simple_tool"}'
    result3 = agent.extract_tool_calls(message3)
    print("Test 3 - Standalone JSON without arguments:")
    print(f"  Input: {message3}")
    print(f"  Result: {result3}")
    print(f"  Success: {len(result3) == 1 and result3[0]['name'] == 'simple_tool'}")
    print()

    # Test 4: Standalone JSON in conversational response
    message4 = 'Let me check the time for you. {"name": "get_current_time", "arguments": {}} I will get that information.'
    result4 = agent.extract_tool_calls(message4)
    print("Test 4 - Standalone JSON in conversation:")
    print(f"  Input: {message4}")
    print(f"  Result: {result4}")
    print(f"  Success: {len(result4) == 1 and result4[0]['name'] == 'get_current_time'}")
    print()

    # Test 5: JSON array (should still work)
    message5 = '[{"name": "tool1", "arguments": {}}, {"name": "tool2", "arguments": {}}]'
    result5 = agent.extract_tool_calls(message5)
    print("Test 5 - JSON array (existing functionality):")
    print(f"  Input: {message5}")
    print(f"  Result: {result5}")
    print(f"  Success: {len(result5) == 2}")
    print()

    # Test 6: Multiple standalone calls in one message
    message6 = 'First call: {"name": "func1", "arguments": {}} and second: {"name": "func2", "arguments": {}}'
    result6 = agent.extract_tool_calls(message6)
    print("Test 6 - Multiple standalone JSON calls:")
    print(f"  Input: {message6}")
    print(f"  Result: {result6}")
    print(f"  Success: {len(result6) == 2}")
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Standalone JSON Tool Call Extraction")
    print("=" * 80)
    print()

    test_standalone_json_extraction()

    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)
