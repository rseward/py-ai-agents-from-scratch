#!/usr/bin/env python

from llama_cpp import Llama
import llama_cpp
import datetime
#from prompt_debugger import PromptDebugger
from memory_manager import MemoryManager
import sys

memory_manager = MemoryManager("./agent-memory.json")

def getSystemPrompt() -> str:
	"""Get the system prompt."""

	memory_summary = memory_manager.get_memory_summary()

	return f"""You are a helpful assistant with long-term memory.
{memory_summary}

When the user shares important information about themselves, their preferences, or facts 
they want you to remember, use the saveMemory function to store it.
"""


def saveMemory(memory_type: str, content: str, key: str = None) -> str:
	"""Save important user information to long term memory for retrieval later (user preferences, facts, etc.)"""

	ret = None
	if memory_type == "fact":
		memory_manager.add_fact(content)
		ret = "Fact saved to memory"
	else:
		memory_manager.add_preference(key, content)
		ret = "Preference saved to memory"

	return datetime.datetime.now().strftime("%I:%M %p")

functions = [saveMemory]
llm = Llama.from_pretrained(
	#repo_id="unsloth/Qwen3-1.7B-GGUF",
	#filename="Qwen3-1.7B-Q8_0.gguf",
	repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
	filename="mistral-7b-instruct-v0.2.Q6_K.gguf",
	functions=functions,
	#chat_format="llama-3",
	echo=True,
	system_prompt=getSystemPrompt(),
    n_ctx=4096*2
)
print("llama.cpp version:", llama_cpp.__version__)

#help(llm.create_chat_completion)
#sys.exit(1 )

def converse(user_prompt: str) -> str:
	"""Converse with the agent."""

	output = llm.create_chat_completion(
		messages = [
			{
				"role": "system",
				"content": getSystemPrompt()
        	},	
			{
				"role": "user",
				"content": user_prompt
			}
		],
		functions=functions
	)

	return output['choices'][0]['message']['content']


# Example conversation
user_prompt=f"""Hi! My name is Alex and I love to program on linux. /nothink"""

print(" ")
print(" ")
print(user_prompt)
print(converse(user_prompt))
print(" ")

user_prompt=f"""What's my favorite activity. /nothink"""

print(" ")
print(" ")
print(user_prompt)
print(converse(user_prompt))
print(" ")
	
#memory_manager.save_memories(memory_manager.get_memories())
