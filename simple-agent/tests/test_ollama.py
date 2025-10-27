#!/usr/bin/env python
"""Minimal test to verify pydantic-ai can connect to Ollama."""

import asyncio
import os
from pydantic_ai import Agent

async def main():
    # Set environment variable
    os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434'

    # Create simple agent
    agent = Agent('ollama:qwen2.5:1.5b')

    # Test query
    result = await agent.run('Say hello in 5 words or less')

    print(f"Result: {result.data}")

if __name__ == "__main__":
    asyncio.run(main())
