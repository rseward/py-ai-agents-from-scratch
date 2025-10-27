#!/usr/bin/env python

from llama_cpp import Llama
import llama_cpp

import sys

# Mixed success with this one. gpt-oss-20b-GGUF spewed non-sense on my CPU and the model 
#  wouldn't load with a reasonable context size to my GPU
# It turns out llama.cpp doesn't support streaming well, if at all. llama-server didn't seem to stream for me.

problem="""What is hoisting in JavaScript? Explain with examples."""

import httpx
import asyncio
import json
import time
import sys

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8123/v1"
MODEL_ALIAS = "Qwen3-1.7B-GGUF:Q4_K_M" # Use the alias reported by your llama-server or defined in your config
BUFFER_FLUSH_INTERVAL_MS = 50  # Flush buffer every 50 milliseconds

# Base template for the request body
BASE_REQUEST_DATA = {
    "model": MODEL_ALIAS,
    "temperature": 0.7,
    "max_tokens": 500,
    "stream": "true" # Enable streaming
}

# --- Time-Buffered Flush Handler ---
async def stream_response_buffer(response_stream):
    """
    Consumes the streaming response, buffers tokens, and flushes
    the buffer to the console on a time interval.
    """
    current_buffer = ""
    last_flush_time = time.time()
    
    # Process the stream asynchronously, line by line
    async for chunk in response_stream.aiter_bytes():
            try:
                # 1. Parse the Server-Sent Event (SSE) data
                decoded_chunk = chunk.decode("utf-8")

                for line in decoded_chunk.splitlines():
                    if line.startswith("data:"):
						# llama-server wasn't streaming well with Qwen3 when I wrote this test
						# so this code block was unused.
                        json_data = line[len("data:"):].strip()
                        if json_data == "[Done]":
                            break
                        try:
                            event_data = json.loads(json_data)
                            # Extract and yeild content
                            if event_data.get("choices"):
                                delta = event_data["choices"].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            pass # Badly formed chunk

                    json_data = json.loads(line)
                    if json_data is not None:
                        # The data likely was sent at the end in it's entirety, so yield it, if it is
						# a well formed json block
                        #print(json_data)
                        yield json_data.get("choices",[])[0].get("message").get("reasoning_content")

            except UnicodeDecodeError:
                # Ignore lines that aren't valid JSON chunks (like [DONE])
                pass
    


# --- Core Asynchronous Request Function ---
async def async_chat_completion_buffered(problem: str):
    """
    Makes a single asynchronous, time-buffered streaming request.
    """
    headers = {"Content-Type": "application/json"}
    
    # 1. Prepare request data
    data = BASE_REQUEST_DATA.copy()
    data['messages'] = [{"role": "user", "content": problem}]

    print(data)

    print(f"🔥 Starting streamed request for problem: '{problem}'")
    print("=" * 80)
    print("LLM Streaming Response:\n")
    
    # Initialize AsyncClient ONCE
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            # 2. Start the streaming POST request
            async with client.stream(
                "POST",
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=data
            ) as response:

                response.raise_for_status() 

                # 3. Process the response stream with the buffered handler
                async for token in stream_response_buffer(response):
                    print(token, end="")

        except httpx.HTTPStatusError as e:
            print(f"\n❌ HTTP Error: {e}")
            print(f"Response text: {e.response.text}")
        except httpx.RequestError as e:
            print(f"\n💥 An error occurred while requesting: {e}")

    print("\n\n" + "=" * 80)


# --- Execution ---
if __name__ == "__main__":
    print(" ")
    print(problem)
    print(" ")
    
    # The output structure is handled entirely within the streaming function, 
    # replacing your original print(output) logic.
    asyncio.run(async_chat_completion_buffered(problem))
