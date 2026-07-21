"""
Web search tool — searches via a self-hosted SearXNG instance.

SearXNG aggregates Google, Bing, DuckDuckGo, and other engines without
per-provider API keys. Point ``searxng_url`` at your instance and call
``GET /search?q=...&format=json``.

Concurrency-safe: multiple searches can run in parallel.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urljoin

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

_DEFAULT_SEARXNG_URL = "http://127.0.0.1:8080"


class WebSearchTool(Tool):
    name = "search_web"
    description = (
        "Search the web using a SearXNG meta-search engine. Returns titles, "
        "URLs, and snippets. Use this to discover relevant pages, then use "
        "fetch_url to read the most promising results in full."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and use relevant keywords.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 10, max: 20).",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(
        self,
        searxng_url: str | None = None,
        default_results: int = 10,
        max_results: int = 20,
        max_result_size_chars: int = 20000,
        http_timeout: int = 30,
        engines: str = "",
        language: str = "auto",
    ):
        self.searxng_url = (
            searxng_url or os.environ.get("SEARXNG_URL") or _DEFAULT_SEARXNG_URL
        ).rstrip("/")
        self.default_results = default_results
        self.max_results = max_results
        self.max_result_size_chars = max_result_size_chars
        self.engines = engines.strip()
        self.language = language.strip() or "auto"
        self._client = httpx.AsyncClient(timeout=float(http_timeout))

    def prompt(self) -> str:
        return (
            "Use search_web to find relevant pages for a topic. Tips:\n"
            "- Use specific, targeted queries (not vague ones)\n"
            "- Try multiple queries with different phrasing for thorough research\n"
            "- Add date qualifiers for time-sensitive topics (e.g., '2024' or 'latest')\n"
            "- After searching, use fetch_url to read the most relevant results"
        )

    def validate_input(self, args: dict) -> ValidationResult:
        query = args.get("query", "")
        if not query or len(query.strip()) < 2:
            return ValidationResult(valid=False, message="Query must be at least 2 characters")
        if len(query) > 500:
            return ValidationResult(valid=False, message="Query too long (max 500 chars)")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        query = args["query"]
        num_results = min(args.get("num_results", self.default_results), self.max_results)

        if context.rate_limiter:
            await context.rate_limiter.acquire(self.searxng_url)

        results = await self._search_searxng(query, num_results)
        if not results:
            return ToolResult(
                data=(
                    f"No search results found for: {query}\n"
                    f"Check that SearXNG is running at {self.searxng_url} "
                    "and that format=json is enabled."
                ),
                is_error=False,
            )

        formatted_parts = [f"## Search Results for: {query}\n"]
        citations: list[Citation] = []

        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("link", result.get("url", ""))
            snippet = result.get("snippet", "No description available")

            formatted_parts.append(f"### {i}. {title}\n**URL**: {url}\n**Snippet**: {snippet}\n")
            if url:
                citations.append(
                    Citation(
                        url=url,
                        title=title,
                        snippet=snippet,
                        source_type=SourceType.WEB,
                    )
                )

        formatted = "\n".join(formatted_parts)
        formatted, truncated, cached_path = await self._maybe_truncate(formatted, query, context)

        return ToolResult(
            data=formatted,
            citations=citations,
            truncated=truncated,
            cached_path=cached_path,
        )

    async def _search_searxng(self, query: str, num_results: int) -> list[dict]:
        """Query SearXNG JSON API and normalize results."""
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "language": self.language,
        }
        if self.engines:
            params["engines"] = self.engines

        endpoint = urljoin(f"{self.searxng_url}/", "search")

        try:
            response = await self._client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"SearXNG search HTTP error ({e.response.status_code}): {e}")
            return []
        except Exception as e:
            logger.error(f"SearXNG search failed: {e}")
            return []

        results: list[dict] = []
        for item in data.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("content") or item.get("snippet") or ""
            if url and title:
                results.append(
                    {
                        "title": title,
                        "link": url,
                        "snippet": snippet,
                        "engine": item.get("engine", ""),
                    }
                )
            if len(results) >= num_results:
                break

        logger.info(f"SearXNG search returned {len(results)} results for '{query[:60]}'")
        return results
