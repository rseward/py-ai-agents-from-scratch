#!/usr/bin/env python
"""Simple memory agent demonstration using AgentWithMemory class with PydanticAI and Ollama."""

import asyncio
from memory_agent import AgentWithMemory


async def main():
    """Main function to demonstrate AgentWithMemory functionality."""

    print("Initializing AgentWithMemory...")
    print("=" * 80)

    # Create AgentWithMemory instance
    agent = AgentWithMemory(
        model_name="mistral:7b-instruct",
        base_url="http://127.0.0.1:11434/v1"
    )

    print("=" * 80)
    print()

    # Test queries
    chat_sequence_1 = [
        "Hi! My name is Alex and I love to program on linux.",
        "What's my favorite activity?"
    ]

    for msg in chat_sequence_1:
        print(f"User: {msg}")
        print("-" * 80)

        # Get response from agent
        response = await agent.chat(msg)

        print(f"Agent: {response}")
        print("=" * 80)
        print()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
