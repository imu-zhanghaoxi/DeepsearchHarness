"""
Post-session memory extraction.

After a completed research session, analyzes the conversation and saves
noteworthy learnings to the persistent memory store.
"""

from __future__ import annotations

import json
import logging

from src.memory.store import MemoryEntry, MemoryStore
from src.memory.types import MemoryType

logger = logging.getLogger(__name__)

_TYPE_MAP = {
    "user": MemoryType.USER,
    "feedback": MemoryType.FEEDBACK,
    "source_reputation": MemoryType.SOURCE_REPUTATION,
    "reference": MemoryType.REFERENCE,
}


async def extract_memories(
    query: str,
    final_answer: str,
    plan_findings: str,
    store: MemoryStore,
) -> list[MemoryEntry]:
    """Extract memorable facts from a completed research session."""
    from src.llm.client import side_query

    try:
        session_text = _build_session_summary(query, final_answer, plan_findings)

        if len(session_text.strip()) < 50:
            logger.debug("Session too short for memory extraction, skipping")
            return []

        existing_headers = await store.get_headers()
        existing_context = ""
        if existing_headers:
            existing_lines = [
                f"- [{h['type']}] {h['title']}: {h['preview']}" for h in existing_headers
            ]
            existing_context = "\n\nExisting memories (do NOT duplicate these):\n" + "\n".join(
                existing_lines
            )

        prompt = f"""Analyze this research session and extract 0-3 noteworthy memories worth saving for future sessions.

## Session
User query: {query}

Final answer (excerpt):
{final_answer[:2000]}

{f"Research findings:{chr(10)}{plan_findings[:1000]}" if plan_findings else ""}
{existing_context}

## Memory types
- "user": User's background, expertise, research interests, preferences
- "feedback": User corrections or preferences about search behavior
- "source_reputation": Which sources were reliable/unreliable for this topic
- "reference": Key facts, dates, or useful URLs discovered during research

## Rules
- Only extract information that would be useful in FUTURE sessions
- Do NOT save raw search results or long passages
- Do NOT duplicate existing memories
- If nothing is worth remembering, return an empty array []
- Keep each memory concise (1-2 sentences)

Return ONLY a JSON array (no other text):
[{{"title": "short title", "content": "concise memory content", "type": "user|feedback|source_reputation|reference"}}]

If nothing worth saving, return: []"""

        response = await side_query(
            prompt=prompt,
            system="Extract persistent memories from research sessions. Return only valid JSON.",
            max_tokens=512,
        )

        if not response or not response.strip():
            logger.debug("Empty response from memory extraction side-query")
            return []

        memories = _parse_memories(response)
        if not memories:
            return []

        saved = []
        for mem_data in memories[:3]:
            title = mem_data.get("title", "").strip()
            content = mem_data.get("content", "").strip()
            type_str = mem_data.get("type", "reference").strip()

            if not title or not content:
                continue

            mem_type = _TYPE_MAP.get(type_str, MemoryType.REFERENCE)
            entry = MemoryEntry(
                title=title,
                content=content,
                memory_type=mem_type,
            )

            path = await store.save(entry)
            saved.append(entry)
            logger.info(f"Memory extracted and saved: [{mem_type.value}] {title} -> {path}")

        if saved:
            logger.info(f"Extracted {len(saved)} memories from session")

        return saved

    except Exception as e:
        logger.warning(f"Memory extraction failed (non-fatal): {e}")
        return []


def _build_session_summary(
    query: str,
    final_answer: str,
    plan_findings: str,
) -> str:
    parts = [f"Query: {query}"]

    if final_answer:
        parts.append(f"Answer: {final_answer[:2000]}")

    if plan_findings:
        parts.append(f"Findings: {plan_findings[:1000]}")

    return "\n\n".join(parts)


def _parse_memories(response: str) -> list[dict]:
    """Parse the LLM's JSON response into a list of memory dicts."""
    text = response.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.debug(f"No JSON array found in extraction response: {text[:100]}")
        return []

    json_text = text[start : end + 1]

    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error in memory extraction: {e}, text: {json_text[:200]}")
        return []
