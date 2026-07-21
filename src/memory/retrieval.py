"""
Relevance-based memory retrieval.

Scans memory headers and uses a side-query to select memories relevant
to the current research question.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.memory.store import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)


async def find_relevant_memories(
    query: str,
    store: MemoryStore,
    max_memories: int = 5,
) -> list[MemoryEntry]:
    """Find memories relevant to the current query."""
    headers = await store.get_headers()

    if not headers:
        return []

    try:
        from src.llm.client import side_query

        headers_text = "\n".join(f"- {h['title']}: {h['preview']}" for h in headers)

        response = await side_query(
            prompt=(f"Query: {query}\n\nAvailable memories:\n{headers_text}"),
            system=(
                "You are selecting memories that will be useful as context for "
                "processing a user's research query. Return a JSON object with a "
                '"selected" field containing a list of memory titles that are '
                "clearly relevant.\n\n"
                "Rules:\n"
                "- Only include memories you are CERTAIN will be helpful\n"
                "- If unsure, do NOT include it — be selective and discerning\n"
                "- If NO memories are relevant, return an empty list\n"
                f"- Maximum {max_memories} memories"
            ),
            max_tokens=256,
            output_schema={
                "type": "object",
                "properties": {
                    "selected": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Titles of relevant memories, or empty list if none are relevant",
                    }
                },
                "required": ["selected"],
                "additionalProperties": False,
            },
        )

        try:
            parsed = json.loads(response)
            selected_titles = parsed.get("selected", [])
        except json.JSONDecodeError:
            selected_titles = _parse_titles_from_text(response, headers)

        if not selected_titles:
            logger.info("Memory selector returned no relevant memories")
            return []

        title_to_header = {h["title"]: h for h in headers}
        entries = []
        for title in selected_titles[:max_memories]:
            header = title_to_header.get(title)
            if header:
                path = Path(header["path"])
                entry = MemoryEntry.from_file(path)
                if entry:
                    entries.append(entry)

        if entries:
            logger.info(
                f"Memory selector picked {len(entries)} relevant memories: "
                + ", ".join(e.title for e in entries)
            )

        return entries

    except Exception as e:
        logger.warning(f"Memory selection failed: {e}")
        return []


def _parse_titles_from_text(response: str, headers: list[dict]) -> list[str]:
    """Fallback: extract memory titles from unstructured text."""
    titles = []
    response_lower = response.lower()
    for h in headers:
        if h["title"].lower() in response_lower:
            titles.append(h["title"])
    return titles


def format_memories_for_prompt(entries: list[MemoryEntry]) -> str:
    """Format memory entries for inclusion in the system prompt."""
    if not entries:
        return ""

    parts = []
    for entry in entries:
        parts.append(f"### [{entry.memory_type.value}] {entry.title}\n{entry.content}\n")

    return "\n".join(parts)
