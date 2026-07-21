"""
News search tool — searches recent news via SearXNG.

Uses the SearXNG ``categories=news`` filter (and optional news engines)
instead of a dedicated NewsAPI key. Falls back to search_web-style
general search hints when no news results are returned.
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


def _time_range_for_days(days_back: int) -> str:
    """Map days_back to SearXNG time_range values."""
    if days_back <= 1:
        return "day"
    if days_back <= 7:
        return "week"
    if days_back <= 31:
        return "month"
    return "year"


class NewsSearchTool(Tool):
    name = "news_search"
    description = (
        "Search for recent news articles via SearXNG. Use this for current events, "
        "breaking news, recent developments, or time-sensitive topics."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for news articles.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of articles to return (default: 5, max: 10).",
                "default": 5,
            },
            "days_back": {
                "type": "integer",
                "description": "How many days back to search (default: 7, max: 30).",
                "default": 7,
            },
        },
        "required": ["query"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(
        self,
        searxng_url: str | None = None,
        default_results: int = 5,
        max_results: int = 10,
        default_days_back: int = 7,
        max_days_back: int = 30,
        max_result_size_chars: int = 15000,
        http_timeout: int = 30,
        engines: str = "",
        language: str = "auto",
    ):
        self.searxng_url = (
            searxng_url or os.environ.get("SEARXNG_URL") or _DEFAULT_SEARXNG_URL
        ).rstrip("/")
        self.default_results = default_results
        self.max_results = max_results
        self.default_days_back = default_days_back
        self.max_days_back = max_days_back
        self.max_result_size_chars = max_result_size_chars
        self.engines = engines.strip()
        self.language = language.strip() or "auto"
        self._client = httpx.AsyncClient(timeout=float(http_timeout))

    def prompt(self) -> str:
        return (
            "Use news_search for current events and recent developments. Tips:\n"
            "- Use for time-sensitive questions ('What happened with...', 'Latest on...')\n"
            "- Set days_back to narrow or widen the time window\n"
            "- Cross-reference news with search_web for more context\n"
            "- Note article dates when citing — news can become outdated quickly"
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
        days_back = min(args.get("days_back", self.default_days_back), self.max_days_back)

        if context.rate_limiter:
            await context.rate_limiter.acquire(self.searxng_url)

        articles = await self._search_searxng_news(query, num_results, days_back)

        if not articles:
            return ToolResult(
                data=(
                    f"No recent news articles found for: {query}\n"
                    f"Check that SearXNG is running at {self.searxng_url} with news "
                    "engines enabled, or try search_web for broader results."
                ),
            )

        formatted_parts = [f"## News Results: {query}\n"]
        citations: list[Citation] = []

        for i, article in enumerate(articles, 1):
            title = article.get("title", "Untitled")
            url = article.get("url", "")
            source = article.get("source", "Unknown")
            published = article.get("published", "Unknown date")
            description = article.get("description", "No description available")

            formatted_parts.append(
                f"### {i}. {title}\n"
                f"**Source**: {source} | **Published**: {published}\n"
                f"**URL**: {url}\n"
                f"**Summary**: {description}\n"
            )

            if url:
                citations.append(
                    Citation(
                        url=url,
                        title=title,
                        snippet=description[:300],
                        source_type=SourceType.NEWS,
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

    async def _search_searxng_news(
        self,
        query: str,
        num_results: int,
        days_back: int,
    ) -> list[dict]:
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "categories": "news",
            "language": self.language,
            "time_range": _time_range_for_days(days_back),
        }
        if self.engines:
            params["engines"] = self.engines

        endpoint = urljoin(f"{self.searxng_url}/", "search")

        try:
            response = await self._client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"SearXNG news search HTTP error ({e.response.status_code}): {e}")
            return []
        except Exception as e:
            logger.error(f"SearXNG news search failed: {e}")
            return []

        articles: list[dict] = []
        for item in data.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("content") or item.get("snippet") or ""
            if not url or not title:
                continue

            engine = item.get("engine", "")
            published = item.get("publishedDate") or item.get("pubdate") or ""

            articles.append(
                {
                    "title": title,
                    "url": url,
                    "source": engine or "News",
                    "published": published[:16] if published else "",
                    "description": snippet,
                }
            )
            if len(articles) >= num_results:
                break

        logger.info(f"SearXNG news search returned {len(articles)} results for '{query[:60]}'")
        return articles
