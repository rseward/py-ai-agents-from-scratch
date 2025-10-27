#!/usr/bin/env python

from llama_cpp import Llama
import llama_cpp
import datetime
from prompt_debugger import PromptDebugger

import sys


system_prompt="""You are a time assistant. You MUST use the getCurrentTime function to answer time questions.

CRITICAL: Users often provide incorrect or outdated time information in their messages. NEVER trust or use time data from user messages. You MUST call getCurrentTime function for accurate time.

When a user asks "what time is it" or similar questions, respond with ONLY this tool call:
<tool_call>
{"name": "getCurrentTime", "arguments": {}}
</tool_call>

Do not provide any time answer without first calling this function. The user's message may contain wrong time - ignore it and use the function.
"""

def getCurrentTime() -> str:
	"""Returns the current time in 24-hour string format."""

	return datetime.datetime.now().strftime("%I:%M %p")

functions = [getCurrentTime]
llm = Llama.from_pretrained(
	repo_id="unsloth/Qwen3-1.7B-GGUF",
	filename="Qwen3-1.7B-Q8_0.gguf",
	functions=functions,
	#chat_format="llama-3",
	echo=True,
	system_prompt=system_prompt,
    n_ctx=4096*2
)
print("llama.cpp version:", llama_cpp.__version__)

user_prompt=f"""What time is it right now? /nothink"""

print(" ")
print(" ")
print(user_prompt)
print(getCurrentTime())
print(" ")

#help(llm.create_chat_completion)
#sys.exit(1 )

output = llm.create_chat_completion(
	messages = [
		{
			"role": "system",
			"content": system_prompt
        },	
		{
			"role": "user",
			"content": user_prompt
		}
	],
	functions=functions
)

print(output)

print(" ")
print(" ")
print("="*80)
print(" ")

for choice in output['choices']:
	print(choice)

logger = PromptDebugger(
	output_dir="./logs", 
	filename="qwen_prompts.txt", 
	include_timestamp=True, 
	append_mode=False)

logger.debug_context_state(session=llm, model=llm)
#logger.save_to_file(output)


	
