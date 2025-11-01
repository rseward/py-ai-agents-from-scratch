"""Comprehensive tests for the MemoryManager class."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_manager import MemoryManager


class TestMemoryManager(unittest.TestCase):
    """Test suite for MemoryManager class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = Path(self.temp_dir) / "test_memory.json"
        self.manager = MemoryManager(str(self.temp_file))

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary file if it exists
        if self.temp_file.exists():
            self.temp_file.unlink()
        # Remove the temporary directory
        Path(self.temp_dir).rmdir()

    def test_init_default_path(self):
        """Test initialization with default path."""
        manager = MemoryManager()
        self.assertEqual(manager.memory_file_path, Path("./memory.json"))

    def test_init_custom_path(self):
        """Test initialization with custom path."""
        custom_path = "/custom/path/memory.json"
        manager = MemoryManager(custom_path)
        self.assertEqual(manager.memory_file_path, Path(custom_path))

    def test_load_memories_empty_file(self):
        """Test loading memories when file doesn't exist."""
        memories = self.manager.load_memories()
        self.assertEqual(memories["facts"], [])
        self.assertEqual(memories["preferences"], {})
        self.assertEqual(memories["conversations"], [])

    def test_load_memories_existing_file(self):
        """Test loading memories from existing file."""
        test_data = {
            "facts": [{"content": "Test fact", "timestamp": "2024-01-01T00:00:00"}],
            "preferences": {"theme": "dark"},
            "conversations": []
        }
        self.temp_file.write_text(json.dumps(test_data))

        memories = self.manager.load_memories()
        self.assertEqual(memories["facts"], test_data["facts"])
        self.assertEqual(memories["preferences"], test_data["preferences"])

    def test_load_memories_invalid_json(self):
        """Test loading memories when file contains invalid JSON."""
        self.temp_file.write_text("invalid json content")

        memories = self.manager.load_memories()
        # Should return empty structure on error
        self.assertEqual(memories["facts"], [])
        self.assertEqual(memories["preferences"], {})
        self.assertEqual(memories["conversations"], [])

    def test_save_memories(self):
        """Test saving memories to file."""
        test_memories = {
            "facts": [{"content": "Test fact", "timestamp": "2024-01-01T00:00:00"}],
            "preferences": {"theme": "dark"},
            "conversations": []
        }

        self.manager.save_memories(test_memories)

        # Verify file was created and contains correct data
        self.assertTrue(self.temp_file.exists())
        loaded_data = json.loads(self.temp_file.read_text())
        self.assertEqual(loaded_data, test_memories)

    def test_save_memories_creates_parent_directory(self):
        """Test that save_memories creates parent directories if needed."""
        nested_path = Path(self.temp_dir) / "nested" / "path" / "memory.json"
        manager = MemoryManager(str(nested_path))

        test_memories = {
            "facts": [],
            "preferences": {},
            "conversations": []
        }

        manager.save_memories(test_memories)
        self.assertTrue(nested_path.exists())

        # Clean up
        nested_path.unlink()
        nested_path.parent.rmdir()
        nested_path.parent.parent.rmdir()

    def test_add_fact(self):
        """Test adding a fact."""
        fact_content = "User prefers Python over JavaScript"
        self.manager.add_fact(fact_content)

        memories = self.manager.load_memories()
        self.assertEqual(len(memories["facts"]), 1)
        self.assertEqual(memories["facts"][0]["content"], fact_content)
        self.assertIn("timestamp", memories["facts"][0])

        # Verify timestamp is valid ISO format
        timestamp = memories["facts"][0]["timestamp"]
        datetime.fromisoformat(timestamp)  # Should not raise

    def test_add_multiple_facts(self):
        """Test adding multiple facts."""
        facts = ["Fact 1", "Fact 2", "Fact 3"]

        for fact in facts:
            self.manager.add_fact(fact)

        memories = self.manager.load_memories()
        self.assertEqual(len(memories["facts"]), 3)

        for i, fact in enumerate(facts):
            self.assertEqual(memories["facts"][i]["content"], fact)

    def test_add_preference(self):
        """Test adding a preference."""
        self.manager.add_preference("theme", "dark")

        memories = self.manager.load_memories()
        self.assertEqual(memories["preferences"]["theme"], "dark")

    def test_add_multiple_preferences(self):
        """Test adding multiple preferences."""
        preferences = {
            "theme": "dark",
            "language": "python",
            "font_size": 14
        }

        for key, value in preferences.items():
            self.manager.add_preference(key, value)

        memories = self.manager.load_memories()
        for key, value in preferences.items():
            self.assertEqual(memories["preferences"][key], value)

    def test_add_preference_overwrite(self):
        """Test that adding a preference overwrites existing value."""
        self.manager.add_preference("theme", "light")
        self.manager.add_preference("theme", "dark")

        memories = self.manager.load_memories()
        self.assertEqual(memories["preferences"]["theme"], "dark")

    def test_add_preference_complex_value(self):
        """Test adding preference with complex value (dict, list)."""
        complex_value = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        self.manager.add_preference("complex", complex_value)

        memories = self.manager.load_memories()
        self.assertEqual(memories["preferences"]["complex"], complex_value)

    def test_get_memory_summary_empty(self):
        """Test getting memory summary with no memories."""
        summary = self.manager.get_memory_summary()

        self.assertIn("=== LONG-TERM MEMORY ===", summary)
        self.assertNotIn("Known Facts:", summary)
        self.assertNotIn("User Preferences:", summary)

    def test_get_memory_summary_with_facts(self):
        """Test getting memory summary with facts."""
        self.manager.add_fact("Python is awesome")
        self.manager.add_fact("Testing is important")

        summary = self.manager.get_memory_summary()

        self.assertIn("=== LONG-TERM MEMORY ===", summary)
        self.assertIn("Known Facts:", summary)
        self.assertIn("- Python is awesome", summary)
        self.assertIn("- Testing is important", summary)

    def test_get_memory_summary_with_preferences(self):
        """Test getting memory summary with preferences."""
        self.manager.add_preference("theme", "dark")
        self.manager.add_preference("language", "python")

        summary = self.manager.get_memory_summary()

        self.assertIn("=== LONG-TERM MEMORY ===", summary)
        self.assertIn("User Preferences:", summary)
        self.assertIn("- theme: dark", summary)
        self.assertIn("- language: python", summary)

    def test_get_memory_summary_complete(self):
        """Test getting memory summary with both facts and preferences."""
        self.manager.add_fact("User is learning Python")
        self.manager.add_preference("theme", "dark")

        summary = self.manager.get_memory_summary()

        self.assertIn("=== LONG-TERM MEMORY ===", summary)
        self.assertIn("Known Facts:", summary)
        self.assertIn("- User is learning Python", summary)
        self.assertIn("User Preferences:", summary)
        self.assertIn("- theme: dark", summary)

    def test_clear_memories(self):
        """Test clearing all memories."""
        self.manager.add_fact("Test fact")
        self.manager.add_preference("theme", "dark")

        self.manager.clear_memories()

        memories = self.manager.load_memories()
        self.assertEqual(memories["facts"], [])
        self.assertEqual(memories["preferences"], {})
        self.assertEqual(memories["conversations"], [])

    def test_get_facts(self):
        """Test getting all facts."""
        self.manager.add_fact("Fact 1")
        self.manager.add_fact("Fact 2")

        facts = self.manager.get_facts()

        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]["content"], "Fact 1")
        self.assertEqual(facts[1]["content"], "Fact 2")

    def test_get_facts_empty(self):
        """Test getting facts when none exist."""
        facts = self.manager.get_facts()
        self.assertEqual(facts, [])

    def test_get_preferences(self):
        """Test getting all preferences."""
        self.manager.add_preference("theme", "dark")
        self.manager.add_preference("language", "python")

        preferences = self.manager.get_preferences()

        self.assertEqual(preferences["theme"], "dark")
        self.assertEqual(preferences["language"], "python")

    def test_get_preferences_empty(self):
        """Test getting preferences when none exist."""
        preferences = self.manager.get_preferences()
        self.assertEqual(preferences, {})

    def test_remove_preference_existing(self):
        """Test removing an existing preference."""
        self.manager.add_preference("theme", "dark")
        self.manager.add_preference("language", "python")

        result = self.manager.remove_preference("theme")

        self.assertTrue(result)
        preferences = self.manager.get_preferences()
        self.assertNotIn("theme", preferences)
        self.assertIn("language", preferences)

    def test_remove_preference_nonexistent(self):
        """Test removing a nonexistent preference."""
        result = self.manager.remove_preference("nonexistent")

        self.assertFalse(result)

    def test_unicode_handling(self):
        """Test handling of unicode characters in facts and preferences."""
        unicode_fact = "User speaks 中文 and русский язык"
        unicode_pref_value = "emoji: 🐍"

        self.manager.add_fact(unicode_fact)
        self.manager.add_preference("unicode", unicode_pref_value)

        memories = self.manager.load_memories()
        self.assertEqual(memories["facts"][0]["content"], unicode_fact)
        self.assertEqual(memories["preferences"]["unicode"], unicode_pref_value)

    def test_persistence_across_instances(self):
        """Test that memories persist across different MemoryManager instances."""
        self.manager.add_fact("Persistent fact")
        self.manager.add_preference("theme", "dark")

        # Create new instance with same file
        new_manager = MemoryManager(str(self.temp_file))
        memories = new_manager.load_memories()

        self.assertEqual(len(memories["facts"]), 1)
        self.assertEqual(memories["facts"][0]["content"], "Persistent fact")
        self.assertEqual(memories["preferences"]["theme"], "dark")

    def test_json_formatting(self):
        """Test that saved JSON is properly formatted with indentation."""
        self.manager.add_fact("Test fact")

        content = self.temp_file.read_text()
        # Check that it's indented (contains newlines and spaces)
        self.assertIn("\n", content)
        self.assertIn("  ", content)

        # Verify it's valid JSON
        parsed = json.loads(content)
        self.assertIsInstance(parsed, dict)


class TestMemoryManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = Path(self.temp_dir) / "test_memory.json"
        self.manager = MemoryManager(str(self.temp_file))

    def tearDown(self):
        """Clean up after each test."""
        if self.temp_file.exists():
            self.temp_file.unlink()
        Path(self.temp_dir).rmdir()

    def test_empty_fact(self):
        """Test adding an empty fact."""
        self.manager.add_fact("")

        memories = self.manager.load_memories()
        self.assertEqual(len(memories["facts"]), 1)
        self.assertEqual(memories["facts"][0]["content"], "")

    def test_empty_preference_key(self):
        """Test adding preference with empty key."""
        self.manager.add_preference("", "value")

        memories = self.manager.load_memories()
        self.assertEqual(memories["preferences"][""], "value")

    def test_none_preference_value(self):
        """Test adding preference with None value."""
        self.manager.add_preference("key", None)

        memories = self.manager.load_memories()
        self.assertIsNone(memories["preferences"]["key"])

    def test_very_long_fact(self):
        """Test adding a very long fact."""
        long_fact = "x" * 10000
        self.manager.add_fact(long_fact)

        memories = self.manager.load_memories()
        self.assertEqual(memories["facts"][0]["content"], long_fact)


if __name__ == "__main__":
    unittest.main()
