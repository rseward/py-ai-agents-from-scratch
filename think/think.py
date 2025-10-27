#!/usr/bin/env python

from llama_cpp import Llama
import llama_cpp

import sys


system_prompt="""You are an expert logical and quantitative reasoner.
    Your goal is to analyze real-world word problems involving families, quantities, averages, and relationships 
    between entities, and compute the exact numeric answer.
    
    Goal: Return the correct final number as a single value — no explanation, no reasoning steps, just the answer.
"""

problem="""My family reunion is this week, and I was assigned the mashed potatoes to bring. 
The attendees include my married mother and father, my twin brother and his family, my aunt and her family, my grandma 
and her brother, her brother's daughter, and his daughter's family. All the adults but me have been married, and no one 
is divorced or remarried, but my grandpa and my grandma's sister-in-law passed away last year. All living spouses are attending. 
My brother has two children that are still kids, my aunt has one six-year-old, and my grandma's brother's daughter has 
three kids under 12. I figure each adult will eat about 1.5 potatoes and each kid will eat about 1/2 a potato, except my 
second cousins don't eat carbs. The average potato is about half a pound, and potatoes are sold in 5-pound bags. 

How many whole bags of potatoes do I need? /nothink
"""

llm = Llama.from_pretrained(
	repo_id="unsloth/Qwen3-1.7B-GGUF",
	filename="Qwen3-1.7B-BF16.gguf",
    n_ctx=4096*2
)
print("llama.cpp version:", llama_cpp.__version__)

print(" ")
print(" ")
print(problem)

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
			"content": problem
		}
	]
)

# 12 Adults * 1.5 = 18 potatoes
# (6 - 3) Kids * 0.5 = 1.5 potatoes
# 18 + 1.5 = 19.5 pounds of potatoes
# 19.5 / 5 = 3.9 bags
# The correct is answer is 4 bags of potatoes
# Qwen3 gives an answer similar to Gemini Flash 2.5

print(output)

print(" ")
print(" ")
print("="*80)
print(" ")

for choice in output['choices']:
	print(choice)
