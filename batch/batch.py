#!/usr/bin/env python

from llama_cpp import Llama
import llama_cpp
import asyncio
import sys

"""
Asynchronous execution improves performance in GAIA benchmarks,
  multi-agent apps and other high-throughput apps.
"""

"""
Per Gemini on 2025-10-26:

Native Batching is for Prompt Processing: The llama-cpp-python library (via the underlying llama.cpp C++
 library) does have a parameter called n_batch. However, this parameter primarily controls the batch size
  used for processing the prompt (the initial input tokens) for a single request, not for generating
   responses for multiple, concurrent prompts.

Sequential Generation: When generating the output tokens, the model typically processes them sequentially
 for each request, which limits the true throughput gains of batching.

The Llama-Server Workaround:

The upstream llama.cpp project includes a built-in llama-server (an HTTP server) that does support parallel
inference for concurrent requests using continuous batching.

To get asynchronous batch processing performance, users often have to run the llama-server binary separately
 and then send concurrent requests to its API endpoints using Python's asyncio and an HTTP client (like httpx).

"""

import httpx
import asyncio
import json

# --- Configuration ---
# The base URL for the llama-server's OpenAI-compatible API
BASE_URL = "http://127.0.0.1:8123/v1"
MODEL_ALIAS = "Qwen3-1.7B-GGUF:Q4_K_M" # Use the alias reported by your llama-server or defined in your config

# Base template for the request body
BASE_REQUEST_DATA = {
    "model": MODEL_ALIAS,
    "temperature": 0.7,
    "max_tokens": 500,
    "stream": False 
}

async def async_chat_completion(client: httpx.AsyncClient, request_message: dict) -> tuple[str, str]:
    """
    Sends a single chat completion request and returns the user content and the response.
    """
    
    headers = {"Content-Type": "application/json"}
    
    # Construct the full request body for this specific chat
    data = BASE_REQUEST_DATA.copy()
    data['messages'] = [request_message]
    
    user_content = request_message.get('content', 'No Content')
    print(f"🚀 Started request for: '{user_content}' to {BASE_URL}/chat/completions")
    
    try:
        response = await client.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=data,
        )
        response.raise_for_status() 

        response_json = response.json()
        
        # Safely extract message content
        message_content = response_json.get('choices', [{}])[0].get('message', {}).get('content', "ERROR: No content in response.")
        
        print(f"✅ Finished request for: '{user_content}'")
        return user_content, message_content
    
    except httpx.HTTPStatusError as e:
        error_msg = f"❌ HTTP Error for '{user_content}': {e}. Response: {e.response.text[:100]}..."
        print(error_msg)
        return user_content, error_msg
    except httpx.RequestError as e:
        error_msg = f"💥 Request Error for '{user_content}': {e}"
        print(error_msg)
        return user_content, error_msg

async def async_chat_completions(requests):
    """
    Makes a series of asynchronous chat completion requests to the llama-server.
    """
    headers = {
        "Content-Type": "application/json",
        # NOTE: If you configured an API key for llama-server, include it here:
        # "Authorization": "Bearer YOUR_LLAMA_API_KEY"
    }

	# Initialize AsyncClient ONCE outside the loop
    async with httpx.AsyncClient(timeout=None) as client:
        # 1. Create a list of all coroutines (async functions) to be run
        tasks = [async_chat_completion(client, req) for req in requests]

        # 2. Use asyncio.gather to run all coroutines concurrently
        print(f"Submitting {len(tasks)} requests concurrently...")
        results = await asyncio.gather(*tasks)

        # 3. Process and print all results once they are all finished
        print("\n" + "="*50)
        print("ALL CONCURRENT REQUESTS COMPLETED")
        print("="*50 + "\n")

        for user_content, response_content in results:
            print(f"Query: {user_content}")
            print("--- Response ---")
            print(response_content.strip())
            print("----------------")
            print("\n" + "-"*50 + "\n")

mychats = [
	{
		"role": "user",
		"content": "Who are you?"
    },
	{
		"role": "user",
		"content": "Hi there, how are you?"
    },	
	{
		"role": "user",
		"content": "How much is 6 * 6? /nothink"
	}
]

# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(async_chat_completions(mychats))




'''
#help(asyncio.run)
#sys.exit(1)

output = asyncio.run(run_single_chat(llm, mychats))

print(output)

print(" ")
print(" ")
print("="*80)
print(" ")

for out in output:
	print(out)
'''
