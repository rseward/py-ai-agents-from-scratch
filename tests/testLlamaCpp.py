#!/usr/bin/env python

from llama_cpp import Llama
import os

#MODEL="../models/Qwen3-Reranker-8B-GGUF/Qwen3-Reranker-8B.Q8_0.gguf"
#MODEL="../models/Qwen3-Reranker-8B-GGUF/Qwen3-Reranker-8B.MXFP4.gguf"
#MODEL="../models/Qwen3-Reranker-8B-GGUF/Qwen3-Reranker-8B.Q5_K_S.gguf"

#assert os.path.exists(MODEL), "Model not found! Did you pull it from huggingface yet?"

# !pip install llama-cpp-python

llm = Llama.from_pretrained(
	repo_id="unsloth/Qwen3-1.7B-GGUF",
	filename="Qwen3-1.7B-BF16.gguf",
)


output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)
