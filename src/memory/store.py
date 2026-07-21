"""
File-based persistent memory store.

Memories are stored as markdown files with YAML frontmatter. A MEMORY.md
index file provides an overview of stored memories.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

from src.memory.types import MemoryType

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""

    title: str
    content: str
    memory_type: MemoryType
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    path: Path | None = None

    def to_frontmatter_md(self) -> str:
        """Serialize to markdown with YAML frontmatter."""
        tags_str = ", ".join(self.tags) if self.tags else ""
        return (
            f"---\n"
            f"title: {self.title}\n"
            f"type: {self.memory_type.value}\n"
            f"tags: [{tags_str}]\n"
            f"created: {self.created_at.isoformat()}\n"
            f"updated: {self.updated_at.isoformat()}\n"
            f"---\n\n"
            f"{self.content}\n"
        )

    @classmethod
    def from_file(cls, path: Path) -> MemoryEntry | None:
        """Parse a memory entry from a markdown file with frontmatter."""
        try:
            text = path.read_text(encoding="utf-8")

            fm_match = re.match(r"^---\n(.*?)\n---\n\n?(.*)", text, re.DOTALL)
            if not fm_match:
                return cls(
                    title=path.stem,
                    content=text,
                    memory_type=MemoryType.REFERENCE,
                    path=path,
                )

            frontmatter = fm_match.group(1)
            content = fm_match.group(2).strip()

            meta: dict[str, str] = {}
            for line in frontmatter.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip()

            tags_str = meta.get("tags", "[]")
            tags = [t.strip() for t in tags_str.strip("[]").split(",") if t.strip()]

            type_str = meta.get("type", "reference")
            try:
                mem_type = MemoryType(type_str)
            except ValueError:
                mem_type = MemoryType.REFERENCE

            return cls(
                title=meta.get("title", path.stem),
                content=content,
                memory_type=mem_type,
                tags=tags,
                created_at=datetime.fromisoformat(meta["created"])
                if "created" in meta
                else datetime.now(),
                updated_at=datetime.fromisoformat(meta["updated"])
                if "updated" in meta
                else datetime.now(),
                path=path,
            )

        except Exception as e:
            logger.warning(f"Failed to parse memory file {path}: {e}")
            return None


class MemoryStore:
    """File-based persistent memory store."""

    def __init__(self, base_dir: str | Path = "./memory"):
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.base_dir / "MEMORY.md"

    async def save(self, entry: MemoryEntry) -> Path:
        """Save a memory entry as a markdown file."""
        if entry.memory_type == MemoryType.USER:
            filename = "user_profile.md"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = re.sub(r"[^\w\s-]", "", entry.title)[:30].strip().replace(" ", "_")
            filename = f"{entry.memory_type.value}_{safe_title}_{ts}.md"

        path = self.base_dir / filename
        entry.path = path
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, partial(path.write_text, entry.to_frontmatter_md(), encoding="utf-8")
        )

        await self._update_index()
        logger.info(f"Memory saved: {path}")
        return path

    async def load_all(self) -> list[MemoryEntry]:
        """Load all memory entries from the base directory."""
        entries = []
        for path in sorted(self.base_dir.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            entry = MemoryEntry.from_file(path)
            if entry:
                entries.append(entry)
        return entries

    async def get_headers(self) -> list[dict]:
        """Get frontmatter headers for all memories."""
        headers = []
        for path in sorted(self.base_dir.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            entry = MemoryEntry.from_file(path)
            if entry:
                headers.append(
                    {
                        "title": entry.title,
                        "type": entry.memory_type.value,
                        "tags": entry.tags,
                        "path": str(entry.path),
                        "updated": entry.updated_at.isoformat(),
                        "preview": entry.content[:100],
                    }
                )
        return headers

    async def _update_index(self) -> None:
        """Rebuild the MEMORY.md index file."""
        entries = await self.load_all()

        if not entries:
            if self.index_path.exists():
                self.index_path.unlink()
            return

        lines = ["# Memory Index\n"]
        by_type: dict[MemoryType, list[MemoryEntry]] = {}
        for entry in entries:
            by_type.setdefault(entry.memory_type, []).append(entry)

        for mem_type, type_entries in by_type.items():
            lines.append(f"\n## {mem_type.value.replace('_', ' ').title()}")
            for entry in type_entries:
                preview = entry.content[:80].replace("\n", " ")
                lines.append(f"- **{entry.title}**: {preview}...")

        lines.append(f"\n---\n*{len(entries)} memories stored*")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, partial(self.index_path.write_text, "\n".join(lines), encoding="utf-8")
        )
