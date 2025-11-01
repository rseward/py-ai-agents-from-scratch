from agent import BaseAgent
from memory_manager import MemoryManager
from pydantic_ai import RunContext
from typing import Optional, List, Dict

class AgentWithMemory(BaseAgent):
    """Agent specialized for with memory tools."""

    def __init__(
        self,
        model_name: str = "mistral:7b-instruct",
        base_url: str = "http://localhost:11434"
    ):
        """Initialize the time agent.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama base URL
        """

        self.memory_manager = MemoryManager("./agent-memory.json")

        # Initialize base agent with time-specific configuration
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            system_prompts=self.prepare_prompt(),
            tools=[]
        )

    def prepare_prompt(self) -> list[str]:
        """Prepare the system prompt for the agent using memories as part of it's input."""

        memory_summary = self.memory_manager.get_memory_summary()
        system_prompts = [ f"""You are a helpful assistant with long-term memory.
{memory_summary}

When the user shares important information about themselves, their preferences, or facts
they want you to remember, use the saveMemory function to store it."""
        ]

        return system_prompts


    def _register_tools(self):
        """Register the get_current_time tool with the agent."""

        # Define the tool function
        def get_current_time() -> str:
            """
            Get the current system time.

            Returns:
                Current time in 12-hour format (HH:MM AM/PM)
            """
            return datetime.datetime.now().strftime("%I:%M %p")

        # Register with PydanticAI agent
        @self.agent.tool
        def saveMemory(ctx: RunContext[None], memory_type: str, content: str, key: str = None) -> str:
            """Save important user information to your long term memory for retrieval later
               (user preferences, facts, etc.)"""

            ret = None
            if memory_type == "fact":
                self.memory_manager.add_fact(content)
                ret = "Fact saved to memory"
            else:
                self.memory_manager.add_preference(key, content)
                ret = "Preference saved to memory"

            return ret

        # Store reference to the tool method for direct access
        self.get_current_time = get_current_time
        self.saveMemory = saveMemory

        # Add to tool mapping for execute_tool_call
        # Map both names to support different calling conventions
        self.tool_mapping['get_current_time'] = get_current_time
        self.tool_mapping['saveMemory'] = saveMemory

    async def chat(self, message: str, message_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Override chat method to inject current memory context into each message.

        Args:
            message: User message
            message_history: Optional message history

        Returns:
            Agent's response as string
        """
        # Get the current memory summary
        memory_summary = self.memory_manager.get_memory_summary()

        # Prepend memory context to the user message
        contextualized_message = f"""[MEMORY CONTEXT]
{memory_summary}

[USER MESSAGE]
{message}"""

        # Call parent's chat method with the contextualized message
        return await super().chat(contextualized_message, message_history)
