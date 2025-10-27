#!/usr/bin/env python
"""Simple time agent demonstration using TimeAgent class with PydanticAI and Ollama."""

import asyncio
from agent import TimeAgent


async def main():
    """Main function to demonstrate TimeAgent functionality."""

    print("Initializing TimeAgent...")
    print("=" * 80)

    # Create TimeAgent instance
    agent = TimeAgent(
        model_name="mistral:7b-instruct",
        base_url="http://127.0.0.1:11434/v1"
    )

    print("=" * 80)
    print()

    # Test queries
    test_queries = [
        "What time is it right now?",
        "The current time is 05:10. What time is it right now?",
        "Can you tell me the current time?",
        "What is the time?"
    ]

    for query in test_queries:
        print(f"User: {query}")
        print("-" * 80)

        # Get response from agent
        response = await agent.chat(query)

        print(f"Agent: {response}")
        print("=" * 80)
        print()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
