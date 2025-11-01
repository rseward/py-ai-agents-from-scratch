"""Memory Manager for storing and retrieving agent memories."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class Fact(TypedDict):
    """Represents a fact stored in memory."""
    content: str
    timestamp: str


class Memory(TypedDict):
    """Represents the complete memory structure."""
    facts: List[Fact]
    preferences: Dict[str, Any]
    conversations: List[Any]


class MemoryManager:
    """Manages persistent memory storage for an agent."""

    def __init__(self, memory_file_path: str = "./memory.json") -> None:
        """Initialize the MemoryManager.

        Args:
            memory_file_path: Path to the JSON file for storing memories.
                Defaults to './memory.json'.
        """
        self.memory_file_path = Path(memory_file_path)

    def load_memories(self) -> Memory:
        """Load memories from the JSON file.

        Returns:
            A dictionary containing facts, preferences, and conversations.
            Returns empty structure if file doesn't exist.
        """
        try:
            data = self.memory_file_path.read_text(encoding="utf-8")
            return json.loads(data)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is invalid, return empty memory
            return {
                "facts": [],
                "preferences": {},
                "conversations": []
            }

    def save_memories(self, memories: Memory) -> None:
        """Save memories to the JSON file.

        Args:
            memories: The memory structure to save.
        """
        # Ensure parent directory exists
        self.memory_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.memory_file_path.write_text(
            json.dumps(memories, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def add_fact(self, fact: str) -> None:
        """Add a specific fact to memory.

        Args:
            fact: The fact content to add.
        """
        memories = self.load_memories()
        memories["facts"].append({
            "content": fact,
            "timestamp": datetime.now().isoformat()
        })
        self.save_memories(memories)

    def add_preference(self, key: str, value: Any) -> None:
        """Add or update a user preference.

        Args:
            key: The preference key.
            value: The preference value.
        """
        memories = self.load_memories()
        memories["preferences"][key] = value
        self.save_memories(memories)

    def get_memory_summary(self) -> str:
        """Get a formatted summary of all memories.

        Returns:
            A formatted string summary of all stored memories.
        """
        memories = self.load_memories()
        summary = "\n=== LONG-TERM MEMORY ===\n"

        if memories["facts"]:
            summary += "\nKnown Facts:\n"
            for fact in memories["facts"]:
                summary += f"- {fact['content']}\n"

        if memories["preferences"]:
            summary += "\nUser Preferences:\n"
            for key, value in memories["preferences"].items():
                summary += f"- {key}: {value}\n"

        return summary

    def clear_memories(self) -> None:
        """Clear all memories by resetting to empty state."""
        empty_memory: Memory = {
            "facts": [],
            "preferences": {},
            "conversations": []
        }
        self.save_memories(empty_memory)

    def get_facts(self) -> List[Fact]:
        """Get all stored facts.

        Returns:
            A list of all facts.
        """
        memories = self.load_memories()
        return memories["facts"]

    def get_preferences(self) -> Dict[str, Any]:
        """Get all stored preferences.

        Returns:
            A dictionary of all preferences.
        """
        memories = self.load_memories()
        return memories["preferences"]

    def remove_preference(self, key: str) -> bool:
        """Remove a preference by key.

        Args:
            key: The preference key to remove.

        Returns:
            True if the preference was removed, False if it didn't exist.
        """
        memories = self.load_memories()
        if key in memories["preferences"]:
            del memories["preferences"][key]
            self.save_memories(memories)
            return True
        return False
