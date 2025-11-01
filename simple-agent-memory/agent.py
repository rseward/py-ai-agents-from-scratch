"""Agent implementation using PydanticAI with Ollama."""

import datetime
import json
import re
import os
from typing import List, Dict, Any, Optional
from pydantic_ai import Agent, RunContext


class BaseAgent:
    """Base agent class with reusable methods for PydanticAI-based agents."""

    def __init__(
        self,
        model_name: str = "mistral:7b-instruct",
        base_url: str = "http://127.0.0.1:11434/v1",
        system_prompts: List[str] = [],
        tools: Optional[List] = None
    ):
        """Initialize the base agent with PydanticAI and Ollama.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama base URL
            system_prompts: System prompt for the agent
            tools: List of tool functions to register
        """
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompts = system_prompts
        self.tools_list = tools or []

        # Build tool mapping from tools list
        self.tool_mapping = {}
        if self.tools_list:
            for tool in self.tools_list:
                if hasattr(tool, '__name__'):
                    self.tool_mapping[tool.__name__] = tool

        # Set Ollama base URL environment variable
        os.environ['OLLAMA_BASE_URL'] = base_url

        # Create the agent instance with Ollama model
        # Using 'ollama:' prefix allows pydantic-ai to auto-detect Ollama
        self.agent = Agent(
            model=f"ollama:{model_name}",
            system_prompt="\n".join(self.system_prompts)
        )

        # Register tools
        self._register_tools()

        print(f"Loaded model: ollama:{model_name} at {base_url}")

    def _register_tools(self):
        """Register tools with the agent. Override in child classes."""
        pass

    def extract_tool_calls(self, message: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from an LLM response.

        Supports various formats:
          - {"name":"tool_name", "arguments": {...}} (standalone JSON)
          - [{"name":"tool_name", "arguments": {...}}]
          - ["tool_name"]({"name":"tool_name","arguments":{...}})
          - <tool_call>{"name":"tool_name", "arguments": {...}}</tool_call>

        Args:
            message: LLM response message

        Returns:
            List of tool call dictionaries with 'name' and 'arguments' keys
        """
        tool_calls = []

        # Preprocess: normalize backtick quotes to regular double quotes
        normalized_message = message.replace('`', '"')

        # Pattern 1: Standalone JSON objects with "name" and optional "arguments"
        # Match: {"name": "tool_name", "arguments": {...}}
        # This pattern looks for JSON objects that are NOT inside arrays or XML tags
        standalone_pattern = r'(?<!\[)\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}(?!\])'
        standalone_matches = re.findall(standalone_pattern, normalized_message)

        for tool_name, args_str in standalone_matches:
            try:
                # Parse the arguments
                arguments = json.loads(args_str) if args_str.strip() != '{}' else {}
                tool_calls.append({
                    'name': tool_name,
                    'arguments': arguments
                })
            except json.JSONDecodeError:
                # If arguments parsing fails, use empty dict
                tool_calls.append({
                    'name': tool_name,
                    'arguments': {}
                })

        # Pattern 2: Standalone JSON without explicit arguments field
        # Match: {"name": "tool_name"}
        simple_standalone_pattern = r'(?<!\[)\s*\{\s*"name"\s*:\s*"([^"]+)"\s*\}(?!\])'
        simple_matches = re.findall(simple_standalone_pattern, normalized_message)

        for tool_name in simple_matches:
            # Only add if not already found by the more complete pattern
            if not any(tc['name'] == tool_name for tc in tool_calls):
                tool_calls.append({
                    'name': tool_name,
                    'arguments': {}
                })

        # Pattern 3: JSON arrays containing tool calls
        # Pattern to match: [{"name":"tool_name", "arguments": {...}}]
        json_pattern = r'\[\s*\{[^\[\]]*"name"[^\[\]]*\}\s*\]'

        matches = re.findall(json_pattern, normalized_message)

        for match in matches:
            try:
                # Parse the JSON
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and 'name' in item:
                            # Ensure arguments exists
                            if 'arguments' not in item:
                                item['arguments'] = {}
                            tool_calls.append(item)
                elif isinstance(parsed, dict) and 'name' in parsed:
                    # Single tool call
                    if 'arguments' not in parsed:
                        parsed['arguments'] = {}
                    tool_calls.append(parsed)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract manually
                continue

        # Pattern 4: Bracket format: ["tool_name"]({"name":"tool_name","arguments":{...}})
        # Pattern matches: ["tool_name"]({JSON object})
        new_format_pattern = r'\["([^"]+)"\]\(\s*(\{.*?\})\s*\)'
        new_matches = re.findall(new_format_pattern, normalized_message)

        for tool_name, json_str in new_matches:
            try:
                # Parse the JSON object
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and 'name' in parsed:
                    # Ensure arguments exists
                    if 'arguments' not in parsed:
                        parsed['arguments'] = {}
                    tool_calls.append(parsed)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a basic call
                tool_calls.append({
                    'name': tool_name,
                    'arguments': {}
                })

        # Pattern 5: XML-style tool calls (Qwen format)
        # <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        xml_matches = re.findall(xml_pattern, normalized_message, re.DOTALL)

        for match in xml_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict) and 'name' in parsed:
                    if 'arguments' not in parsed:
                        parsed['arguments'] = {}
                    tool_calls.append(parsed)
            except json.JSONDecodeError:
                continue

        return tool_calls

    def execute_tool_call(self, callinfo: Dict[str, Any]) -> str:
        """
        Execute a tool call given its dictionary representation.

        Args:
            callinfo: Dictionary with 'name' and 'arguments' keys

        Returns:
            Result of the tool execution as a string
        """
        if not isinstance(callinfo, dict) or 'name' not in callinfo:
            return "Error: Invalid tool call format"

        tool_name = callinfo['name']
        arguments = callinfo.get('arguments', {})

        if tool_name not in self.tool_mapping:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            tool_function = self.tool_mapping[tool_name]

            # Execute the tool with provided arguments
            result = tool_function(**arguments)
            return str(result)

        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    async def completion(self, message: str, role: str = "user", debug: bool = False, message_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Send a message and get completion, handling tool calls with proper message history.

        Args:
            message: User message
            role: Message role (default: "user")
            debug: Enable debug output
            message_history: Optional message history to continue conversation

        Returns:
            Dictionary containing:
                - response: Final response text
                - history: Updated message history including tool calls
                - tool_calls_executed: List of tool calls that were executed
        """
        # Initialize message history if not provided
        if message_history is None:
            message_history = []

        # Add user message to history
        message_history.append({
            "role": role,
            "content": message
        })

        # Run the agent
        result = await self.agent.run(message)

        # Handle different result types from PydanticAI
        lastmsg = None
        if hasattr(result, 'data') and result.data is not None:
            lastmsg = str(result.data)
        elif hasattr(result, 'output') and result.output is not None:
            lastmsg = str(result.output).strip()
        else:
            lastmsg = str(result)

        if debug:
            print(f"[DEBUG] Initial response: {lastmsg}")

        # Add assistant's initial response to history
        message_history.append({
            "role": "assistant",
            "content": lastmsg
        })

        # Extract and execute tool calls
        tool_calls = self.extract_tool_calls(lastmsg)
        tool_calls_executed = []

        if debug:
            print(f"[DEBUG] Extracted tool_calls: {tool_calls}")

        if tool_calls:
            # Execute each tool call and add results to history
            for tool_call in tool_calls:
                if debug:
                    print(f"[DEBUG] Executing tool_call: {tool_call}")

                tool_result = self.execute_tool_call(tool_call)

                if debug:
                    print(f"[DEBUG] Tool result: {tool_result}")

                # Add tool result to message history with role "tool"
                message_history.append({
                    "role": "tool",
                    "name": tool_call.get('name', 'unknown'),
                    "content": tool_result
                })

                tool_calls_executed.append({
                    'call': tool_call,
                    'result': tool_result
                })

            # Build a message with all tool results to send back to the model
            tool_results_text = "\n".join([
                f"Tool '{tc['call']['name']}' returned: {tc['result']}"
                for tc in tool_calls_executed
            ])

            follow_up_message = f"{tool_results_text}\n\nNow provide your response to the user based on these tool results."

            if debug:
                print(f"[DEBUG] Sending follow-up message to model:\n{follow_up_message}")

            # Get final response from model with tool results
            final_result = await self.agent.run(follow_up_message)

            final_response = None
            if hasattr(final_result, 'data') and final_result.data is not None:
                final_response = str(final_result.data)
            elif hasattr(final_result, 'output') and final_result.output is not None:
                final_response = str(final_result.output).strip()
            else:
                final_response = str(final_result)

            if debug:
                print(f"[DEBUG] Final response: {final_response}")

            # Add final response to history
            message_history.append({
                "role": "assistant",
                "content": final_response
            })

            return {
                'response': final_response,
                'history': message_history,
                'tool_calls_executed': tool_calls_executed
            }

        # No tool calls, return original response
        return {
            'response': lastmsg,
            'history': message_history,
            'tool_calls_executed': []
        }

    def is_ollama_available(self) -> bool:
        """
        Check if Ollama is available and the model is accessible.

        Returns:
            True if Ollama and model are available
        """
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                # Check if our model is in the list
                for model in models:
                    if model.get('name', '').startswith(self.model_name):
                        return True
            return False
        except Exception:
            return False

    async def run(self, message: str, message_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Run the agent with a message (wrapper for completion).

        Args:
            message: User message
            message_history: Optional message history

        Returns:
            Dictionary with response, history, and tool_calls_executed
        """
        return await self.completion(message, message_history=message_history)

    async def chat(self, message: str, message_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Send a message to the agent and get a response.

        Args:
            message: User message
            message_history: Optional message history

        Returns:
            Agent's response as string
        """
        try:
            result = await self.completion(message, message_history=message_history)
            return result['response']
        except Exception as e:
            return f"Error: {str(e)}"


