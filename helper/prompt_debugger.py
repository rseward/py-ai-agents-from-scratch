"""
PromptDebugger for python-llama-cpp Chat Sessions

A debugging utility to capture and analyze prompts, context states, and token structures
from python-llama-cpp chat sessions. Useful for debugging conversation flow, token usage,
and prompt formatting.

Basic Usage:
    # Initialize debugger
    debugger = PromptDebugger(
        output_dir='./debug_logs',
        filename='chat_debug.txt',
        output_types=[OutputTypes.EXACT_PROMPT, OutputTypes.CONTEXT_STATE]
    )
    
    # Capture and log to console
    data = debugger.log_to_console(
        session=chat_session,
        prompt="What is the weather like?",
        system_prompt="You are a helpful assistant.",
        functions={'get_weather': weather_func}
    )
    
    # Save to file
    result = debugger.debug(
        session=chat_session,
        model=llama_model,
        prompt="What is the weather like?"
    )
    debugger.save_to_file(result['captured_data'])

Quick Functions:
    # Debug only exact prompt (minimal requirements)
    result = debug_exact_prompt(
        session=chat_session,
        prompt="Hello world"
    )

    # Debug full context state (requires model)
    result = debug_context_state(
        session=chat_session,
        model=llama_model
    )

    # Debug token structure
    result = debug_structured(
        session=chat_session,
        model=llama_model
    )

    # Debug everything
    result = debug_all(
        session=chat_session,
        model=llama_model,
        prompt="Hello world"
    )

Output Types:
    - EXACT_PROMPT: Shows the formatted prompt sent to the model
    - CONTEXT_STATE: Shows the full conversation context with responses
    - STRUCTURED: Shows detailed token-level breakdown

Parameters:
    - session: The python-llama-cpp chat session object (required)
    - model: The loaded Llama model (required for CONTEXT_STATE and STRUCTURED)
    - prompt: The user's input text (required for EXACT_PROMPT)
    - system_prompt: Optional system prompt
    - functions: Optional dictionary of available functions
"""

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class OutputTypes(Enum):
    EXACT_PROMPT = 'exactPrompt'
    CONTEXT_STATE = 'contextState'
    STRUCTURED = 'structured'


class PromptDebugger(object):
    """Helper class for debugging and logging LLM prompts for python-llama-cpp"""
    
    def __init__(self, 
                 output_dir: str = './',
                 filename: Optional[str] = None,
                 include_timestamp: bool = False,
                 append_mode: bool = False,
                 output_types: Optional[List[OutputTypes]] = None):
        self.output_dir = output_dir
        self.filename = filename
        self.include_timestamp = include_timestamp
        self.append_mode = append_mode
        
        # Configure which outputs to include
        if output_types is None:
            output_types = [OutputTypes.EXACT_PROMPT]
        self.output_types = output_types
        
        # Ensure output_types is always a list
        if not isinstance(self.output_types, list):
            self.output_types = [self.output_types]
    
    def capture_exact_prompt(self, 
                            session: Any,
                            prompt: str,
                            system_prompt: Optional[str] = None,
                            functions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Captures only the exact prompt (user input + system + functions)
        
        Args:
            session: The chat session
            prompt: The user prompt
            system_prompt: System prompt (optional)
            functions: Available functions (optional)
                
        Returns:
            Dict containing the exact prompt data
        """
        
        chat_wrapper = session.chat_wrapper
        
        # Build minimal history for exact prompt
        history = [{'type': 'user', 'text': prompt}]
        
        if system_prompt:
            history.insert(0, {'type': 'system', 'text': system_prompt})
        
        # Generate the context state with just the current prompt
        state = chat_wrapper.generate_context_state(
            chat_history=history,
            available_functions=functions,
            system_prompt=system_prompt
        )
        
        formatted_prompt = str(state.context_text)
        
        return {
            'exact_prompt': formatted_prompt,
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'system_prompt': system_prompt,
            'functions': list(functions.keys()) if functions else []
        }
    
    def capture_context_state(self,
                             session: Any,
                             model: Any) -> Dict[str, Any]:
        """
        Captures the full context state (includes assistant responses)

        Args:
            session: The chat session (llama model object)
            model: The loaded model

        Returns:
            Dict containing the context state data
        """

        # Get the actual context from the session after responses
        # Handle both old API (session.sequence.context_tokens) and new API (model.input_ids)
        if hasattr(session, 'sequence') and hasattr(session.sequence, 'context_tokens'):
            context_tokens = session.sequence.context_tokens
        elif hasattr(session, 'input_ids'):
            context_tokens = session.input_ids
        else:
            raise AttributeError("Cannot find context tokens in session object")

        context_state = model.detokenize(context_tokens.tolist() if hasattr(context_tokens, 'tolist') else list(context_tokens), special=True)

        # detokenize returns bytes, so decode to string
        if isinstance(context_state, bytes):
            context_state = context_state.decode('utf-8', errors='replace')

        return {
            'context_state': context_state,
            'timestamp': datetime.now().isoformat(),
            'token_count': len(context_tokens)
        }
    
    def capture_structured(self,
                          session: Any,
                          model: Any) -> Dict[str, Any]:
        """
        Captures the structured token representation

        Args:
            session: The chat session (llama model object)
            model: The loaded model

        Returns:
            Dict containing the structured token data
        """

        # For python-llama-cpp, we'll use the tokenizer to get structured data
        # Handle both old API (session.sequence.context_tokens) and new API (model.input_ids)
        if hasattr(session, 'sequence') and hasattr(session.sequence, 'context_tokens'):
            tokens = session.sequence.context_tokens
        elif hasattr(session, 'input_ids'):
            tokens = session.input_ids
        else:
            raise AttributeError("Cannot find context tokens in session object")

        # Decode tokens to strings, handling bytes return type
        token_strings = []
        for token in tokens:
            decoded = model.tokenizer.decode([token])
            if isinstance(decoded, bytes):
                decoded = decoded.decode('utf-8', errors='replace')
            token_strings.append(decoded)

        structured = {
            'tokens': tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens),
            'token_strings': token_strings,
            'length': len(tokens)
        }

        return {
            'structured': structured,
            'timestamp': datetime.now().isoformat(),
            'token_count': len(tokens)
        }
    
    def capture_all(self, 
                   session: Any,
                   model: Optional[Any] = None,
                   prompt: Optional[str] = None,
                   system_prompt: Optional[str] = None,
                   functions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Captures all configured output types
        
        Args:
            session: The chat session
            model: The loaded model (required for CONTEXT_STATE and STRUCTURED)
            prompt: The user prompt (required for EXACT_PROMPT)
            system_prompt: Optional system prompt
            functions: Optional dictionary of available functions
            
        Returns:
            Dict containing combined captured data based on configuration
        """
        result = {
            'timestamp': datetime.now().isoformat()
        }
        
        if OutputTypes.EXACT_PROMPT in self.output_types:
            if prompt is None:
                raise ValueError("prompt is required for EXACT_PROMPT output type")
            exact_data = self.capture_exact_prompt(session, prompt, system_prompt, functions)
            result['exact_prompt'] = exact_data['exact_prompt']
            result['prompt'] = exact_data['prompt']
            result['system_prompt'] = exact_data['system_prompt']
            result['functions'] = exact_data['functions']
        
        if OutputTypes.CONTEXT_STATE in self.output_types:
            if model is None:
                raise ValueError("model is required for CONTEXT_STATE output type")
            context_data = self.capture_context_state(session, model)
            result['context_state'] = context_data['context_state']
            result['context_token_count'] = context_data['token_count']
        
        if OutputTypes.STRUCTURED in self.output_types:
            if model is None:
                raise ValueError("model is required for STRUCTURED output type")
            structured_data = self.capture_structured(session, model)
            result['structured'] = structured_data['structured']
            result['structured_token_count'] = structured_data['token_count']
        
        return result
    
    def format_output(self, captured_data: Dict[str, Any]) -> str:
        """
        Formats the captured data based on configuration
        
        Args:
            captured_data: Data from capture methods
            
        Returns:
            Formatted output string
        """
        output = "\n========== PROMPT DEBUG OUTPUT ==========\n"
        output += f"Timestamp: {captured_data['timestamp']}\n"
        
        if 'prompt' in captured_data and captured_data['prompt']:
            output += f"Original Prompt: {captured_data['prompt']}\n"
        
        if 'system_prompt' in captured_data and captured_data['system_prompt']:
            system_preview = captured_data['system_prompt'][:50] + "..." if len(captured_data['system_prompt']) > 50 else captured_data['system_prompt']
            output += f"System Prompt: {system_preview}\n"
        
        if 'functions' in captured_data and captured_data['functions']:
            output += f"Functions: {', '.join(captured_data['functions'])}\n"
        
        if 'exact_prompt' in captured_data:
            output += "\n=== EXACT PROMPT ===\n"
            output += captured_data['exact_prompt']
            output += "\n"
        
        if 'context_state' in captured_data:
            output += f"Token Count: {captured_data.get('context_token_count', 'N/A')}\n"
            output += "\n=== CONTEXT STATE ===\n"
            output += captured_data['context_state']
            output += "\n"
        
        if 'structured' in captured_data:
            output += "\n=== STRUCTURED ===\n"
            output += f"Token Count: {captured_data.get('structured_token_count', 'N/A')}\n"
            output += json.dumps(captured_data['structured'], indent=2)
            output += "\n"
        
        output += "==========================================\n"
        return output
    
    def save_to_file(self, captured_data: Dict[str, Any], custom_filename: Optional[str] = None) -> str:
        """
        Saves data to file

        Args:
            captured_data: Data to save
            custom_filename: Optional custom filename

        Returns:
            Path to the saved file
        """
        content = self.format_output(captured_data)
        
        filename = custom_filename or self.filename
        
        if self.include_timestamp:
            timestamp = datetime.now().isoformat().replace(':', '-').replace('.', '-')
            filepath = Path(filename)
            base = filepath.stem
            ext = filepath.suffix
            filename = f"{base}_{timestamp}{ext}"
        
        filepath = Path(self.output_dir) / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if self.append_mode else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(content)
        
        print(f"Prompt debug output written to {filepath}")
        return str(filepath)
    
    def save_to_file_sync(self, captured_data: Dict[str, Any], custom_filename: Optional[str] = None) -> str:
        """
        Synchronous version of save_to_file
        """
        content = self.format_output(captured_data)
        
        filename = custom_filename or self.filename
        
        if self.include_timestamp:
            timestamp = datetime.now().isoformat().replace(':', '-').replace('.', '-')
            filepath = Path(filename)
            base = filepath.stem
            ext = filepath.suffix
            filename = f"{base}_{timestamp}{ext}"
        
        filepath = Path(self.output_dir) / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if self.append_mode else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(content)
        
        print(f"Prompt debug output written to {filepath}")
        return str(filepath)
    
    def debug_exact_prompt(self,
                               session: Any,
                               prompt: str,
                               system_prompt: Optional[str] = None,
                               functions: Optional[Dict[str, Any]] = None,
                               custom_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Debug exact prompt only - minimal params needed

        Args:
            session: The chat session
            prompt: The user prompt
            system_prompt: Optional system prompt
            functions: Optional dictionary of available functions
            custom_filename: Optional custom filename

        Returns:
            Dict containing captured_data and filepath
        """
        old_output_types = self.output_types.copy()
        self.output_types = [OutputTypes.EXACT_PROMPT]
        captured_data = self.capture_all(session=session, prompt=prompt,
                                       system_prompt=system_prompt, functions=functions)
        filepath = self.save_to_file(captured_data, custom_filename)
        self.output_types = old_output_types
        return {'captured_data': captured_data, 'filepath': filepath}
    
    def debug_context_state(self,
                                session: Any,
                                model: Any,
                                custom_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Debug context state only - needs session and model

        Args:
            session: The chat session
            model: The loaded model
            custom_filename: Optional custom filename

        Returns:
            Dict containing captured_data and filepath
        """
        old_output_types = self.output_types.copy()
        self.output_types = [OutputTypes.CONTEXT_STATE]
        captured_data = self.capture_all(session=session, model=model)
        filepath = self.save_to_file(captured_data, custom_filename)
        self.output_types = old_output_types
        return {'captured_data': captured_data, 'filepath': filepath}
    
    def debug_structured(self,
                             session: Any,
                             model: Any,
                             custom_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Debug structured only - needs session and model

        Args:
            session: The chat session
            model: The loaded model
            custom_filename: Optional custom filename

        Returns:
            Dict containing captured_data and filepath
        """
        old_output_types = self.output_types.copy()
        self.output_types = [OutputTypes.STRUCTURED]
        captured_data = self.capture_all(session=session, model=model)
        filepath = self.save_to_file(captured_data, custom_filename)
        self.output_types = old_output_types
        return {'captured_data': captured_data, 'filepath': filepath}
    
    def debug(self,
                  session: Any,
                  model: Optional[Any] = None,
                  prompt: Optional[str] = None,
                  system_prompt: Optional[str] = None,
                  functions: Optional[Dict[str, Any]] = None,
                  custom_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Debug with configured output types

        Args:
            session: The chat session
            model: The loaded model (required for CONTEXT_STATE and STRUCTURED)
            prompt: The user prompt (required for EXACT_PROMPT)
            system_prompt: Optional system prompt
            functions: Optional dictionary of available functions
            custom_filename: Optional custom filename

        Returns:
            Dict containing captured_data
        """
        captured_data = self.capture_all(session=session, model=model, prompt=prompt,
                                       system_prompt=system_prompt, functions=functions)
        return {'captured_data': captured_data}
    
    def log_to_console(self, 
                      session: Any,
                      model: Optional[Any] = None,
                      prompt: Optional[str] = None,
                      system_prompt: Optional[str] = None,
                      functions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log to console only
        
        Args:
            session: The chat session
            model: The loaded model (required for CONTEXT_STATE and STRUCTURED)
            prompt: The user prompt (required for EXACT_PROMPT)
            system_prompt: Optional system prompt
            functions: Optional dictionary of available functions
            
        Returns:
            Captured data
        """
        captured_data = self.capture_all(session=session, model=model, prompt=prompt,
                                       system_prompt=system_prompt, functions=functions)
        print(self.format_output(captured_data))
        return captured_data
    
    def log_exact_prompt(self, 
                        session: Any,
                        prompt: str,
                        system_prompt: Optional[str] = None,
                        functions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Log exact prompt to console"""
        captured_data = self.capture_exact_prompt(session, prompt, system_prompt, functions)
        print(self.format_output(captured_data))
        return captured_data
    
    def log_context_state(self, 
                         session: Any,
                         model: Any) -> Dict[str, Any]:
        """Log context state to console"""
        captured_data = self.capture_context_state(session, model)
        print(self.format_output(captured_data))
        return captured_data
    
    def log_structured(self, 
                      session: Any,
                      model: Any) -> Dict[str, Any]:
        """Log structured to console"""
        captured_data = self.capture_structured(session, model)
        print(self.format_output(captured_data))
        return captured_data


# Quick functions for convenience
def debug_exact_prompt(session: Any,
                            prompt: str,
                            system_prompt: Optional[str] = None,
                            functions: Optional[Dict[str, Any]] = None,
                            **kwargs) -> Dict[str, Any]:
    """Quick function to debug exact prompt only"""
    prompt_debugger = PromptDebugger(
        output_types=[OutputTypes.EXACT_PROMPT],
        **kwargs
    )
    return prompt_debugger.debug(session=session, prompt=prompt,
                                     system_prompt=system_prompt, functions=functions)


def debug_context_state(session: Any,
                             model: Any,
                             **kwargs) -> Dict[str, Any]:
    """Quick function to debug context state only"""
    prompt_debugger = PromptDebugger(
        output_types=[OutputTypes.CONTEXT_STATE],
        **kwargs
    )
    return prompt_debugger.debug(session=session, model=model)


def debug_structured(session: Any,
                          model: Any,
                          **kwargs) -> Dict[str, Any]:
    """Quick function to debug structured only"""
    prompt_debugger = PromptDebugger(
        output_types=[OutputTypes.STRUCTURED],
        **kwargs
    )
    return prompt_debugger.debug(session=session, model=model)


def debug_all(session: Any,
                   model: Optional[Any] = None,
                   prompt: Optional[str] = None,
                   system_prompt: Optional[str] = None,
                   functions: Optional[Dict[str, Any]] = None,
                   **kwargs) -> Dict[str, Any]:
    """Quick function to debug all outputs"""
    prompt_debugger = PromptDebugger(
        output_types=[OutputTypes.EXACT_PROMPT, OutputTypes.CONTEXT_STATE, OutputTypes.STRUCTURED],
        **kwargs
    )
    return prompt_debugger.debug(
        session=session, 
        model=model, 
        prompt=prompt,
        system_prompt=system_prompt, 
        functions=functions
        )