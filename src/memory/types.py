"""
Memory type taxonomy.

Defines what kinds of memories the system can store and retrieve.
"""

from __future__ import annotations

from enum import Enum


class MemoryType(str, Enum):
    """Types of persistent memories the system stores."""

    USER = "user"
    FEEDBACK = "feedback"
    SOURCE_REPUTATION = "source_reputation"
    REFERENCE = "reference"
