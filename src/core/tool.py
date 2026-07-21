from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.types import ToolResult, ValidationResult

logger = logging.getLogger(__name__)
CACHE_DIR = Path("./cache")

@dataclass
class ToolUseContext:
    session_id: str = ""
    turn_count: int = 0
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    # Rate limiter reference (injected by the loop)
    rate_limiter: Any = None
    # Extra context tools might need
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

class Tool(ABC):
    name: str = ""
    description: str = ""
    input_schema: dict

    is_concurrency_safe: bool = False
    is_read_only: bool = True
    max_result_size_chars: int = 50000

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "input_schema" not in cls.__dict__:
            cls.input_schema = {}
            
    @abstractmethod
    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        pass

    def prompt(self) -> str:
        return f"You are using the {self.name} tool. {self.description}"

    def validate_input(self, args: dict) -> ValidationResult:
        return ValidationResult(valid=True)
        
    async def aclose(self) -> None:
        """
        Close any resources held by this tool (e.g., httpx clients).

        Called during application shutdown. Subclasses with HTTP clients
        don't need to override this — the default implementation closes
        any httpx.AsyncClient found on standard attribute names.
        """
        for attr in ("_client", "_jina_client"):
            client = getattr(self, attr, None)
            if client is not None and hasattr(client, "aclose"):
                try:
                    await client.aclose()
                except Exception:
                    pass

    def to_api_schema(self) -> dict:
        """
        Convert to the format expected by the OpenAI function calling API.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    async def _maybe_truncate(self, data: str, url: str, context: ToolUseContext) -> tuple[str, bool, str | None]:
        """
        If data exceeds max_result_size_chars, cache to disk and return preview.

        Oversized results are persisted to disk with a preview + path
        in the context, so the agent can use deep_read to access them.
        """
        if len(data) <= self.max_result_size_chars:
            return data, False, None

        # Cache full content to disk
        import hashlib
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cached_path = context.cache_dir / f"{self.name}_{url_hash}.md"
        cached_path.write_text(data, encoding="utf-8")

        # Return truncated preview
        preview = data[: self.max_result_size_chars]
        preview += f"\n\n---\n[Content truncated. Full content ({len(data):,} chars) saved to: {cached_path}]"

        logger.info(f"Tool {self.name}: cached {len(data):,} chars to {cached_path}")
        return preview, True, str(cached_path)


class ToolRegistry:
    """
    Collects all available tools and provides lookup.

    Single source of truth for which tools are available. Tools are
    registered at startup and looked up by name during the agentic loop.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Last registration wins on name collision."""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def remove(self, name: str) -> None:
        """Remove a tool by name (no-op if not registered)."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Removed tool: {name}")

    def all_tools(self) -> list[Tool]:
        """All registered tools (for system prompt building)."""
        return list(self._tools.values())

    def get_api_schemas(self) -> list[dict]:
        """All tool schemas for LLM function calling API."""
        return [tool.to_api_schema() for tool in self._tools.values()]

    def get_concurrent_safe(self) -> set[str]:
        """Names of tools that can run in parallel."""
        return {name for name, tool in self._tools.items() if tool.is_concurrency_safe}

    async def close_all(self) -> None:
        """Close any HTTP clients held by registered tools."""
        for tool in self._tools.values():
            try:
                await tool.aclose()
            except Exception:
                pass
        
def build_default_registry(config: dict | None = None) -> ToolRegistry:
    """
    Build the default tool registry with all search tools.

    Called once at startup.

    Args:
        config: Optional dict of tool configuration from settings.yaml.
                Keys like "web_search_default_results", "searxng_url", etc.
    """
    from src.tools.web_fetch import WebFetchTool
    from src.tools.web_search import WebSearchTool

    cfg = config or {}
    registry = ToolRegistry()
    registry.register(
        WebSearchTool(
            searxng_url=cfg.get("searxng_url", ""),
            default_results=cfg.get("web_search_default_results", 10),
            max_results=cfg.get("web_search_max_results", 20),
            max_result_size_chars=cfg.get("max_result_size_chars", 20000),
            http_timeout=cfg.get("http_timeout", 30),
            engines=cfg.get("searxng_engines", ""),
            language=cfg.get("searxng_language", "auto"),
        )
    )
    registry.register(
        WebFetchTool(
            max_result_size_chars=cfg.get("max_result_size_chars", 50000),
            http_timeout=cfg.get("http_timeout", 30),
            jina_timeout=cfg.get("jina_timeout", 60),
            extraction_threshold=cfg.get("content_extraction_threshold", 15000),
        )
    )
    return registry

