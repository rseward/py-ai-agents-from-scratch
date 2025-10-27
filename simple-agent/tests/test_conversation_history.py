#!/usr/bin/env python
"""Test conversation with message history across multiple turns."""

import asyncio
from agent import TimeAgent


async def main():
    """Test multi-turn conversation with message history."""

    print("=" * 80)
    print("Testing Multi-Turn Conversation with Message History")
    print("=" * 80)
    print()

    # Create TimeAgent instance
    agent = TimeAgent(
        model_name="mistral:7b-instruct",
        base_url="http://127.0.0.1:11434/v1"
    )

    print("Initialized TimeAgent")
    print()

    # Start with empty history
    history = []

    # Turn 1: Ask for the time
    print("Turn 1:")
    print("-" * 80)
    query1 = "What time is it?"
    print(f"User: {query1}")

    result1 = await agent.run(query1, message_history=history)
    history = result1['history']

    print(f"Agent: {result1['response']}")
    print(f"Tool calls executed: {len(result1['tool_calls_executed'])}")
    print(f"History length: {len(history)} messages")
    print()

    # Turn 2: Follow-up question (should use history)
    print("Turn 2:")
    print("-" * 80)
    query2 = "Thanks! Can you tell me again?"
    print(f"User: {query2}")

    result2 = await agent.run(query2, message_history=history)
    history = result2['history']

    print(f"Agent: {result2['response']}")
    print(f"Tool calls executed: {len(result2['tool_calls_executed'])}")
    print(f"History length: {len(history)} messages")
    print()

    # Display complete conversation history
    print("=" * 80)
    print("COMPLETE CONVERSATION HISTORY:")
    print("=" * 80)
    for i, msg in enumerate(history, 1):
        role = msg['role'].upper()
        content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        if msg['role'] == 'tool':
            print(f"{i}. [{role} - {msg.get('name', 'unknown')}] {content}")
        else:
            print(f"{i}. [{role}] {content}")
    print()

    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
