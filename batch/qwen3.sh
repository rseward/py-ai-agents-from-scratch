#!/bin/bash

# brew install llama.cpp

llama-server --threads 4 --host 0.0.0.0 --port 8123 --alias Qwen3-1.7B-GGUF:Q4_K_M -hf unsloth/Qwen3-1.7B-GGUF:Q4_K_M
