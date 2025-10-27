#!/usr/bin/env python

from llama_cpp import Llama

import sys


llm = Llama.from_pretrained(
	repo_id="unsloth/Qwen3-1.7B-GGUF",
	filename="Qwen3-1.7B-BF16.gguf",
)
print("llama.cpp version:", llm.get_version())
sys.exit(1)


output = llm(
      "Q: Do you know node-llama-cpp? \n A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(dir(output))
print(" ")
print(" ")
print("="*80)
print(" ")

print(output['choices'][0]['text'])
