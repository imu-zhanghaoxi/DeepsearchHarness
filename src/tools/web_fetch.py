"""
Web fetch tool — fetches a URL and converts it to readable markdown.

Fetch strategies (tried in order):
1. Jina Reader API (r.jina.ai) — markdown directly, handles JS/PDF better.
2. Direct fetch + trafilatura/markdownify — no API key, weaker on JS pages.

Concurrency-safe: multiple pages can be fetched in parallel.
"""

from __future__ import annotations

import logging
import os
import re

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult
from src.utils.url_validator import validate_url_for_ssrf, validate_url_for_ssrf_async

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

JINA_READER_URL = "https://r.jina.ai/"

_SEARCH_ENGINE_URL_MARKERS = (
    "duckduckgo.com",
    "google.com/search",
    "bing.com/search",
    "search.yahoo.com",
    "yandex.com/search",
    "baidu.com/s",
)


def _html_to_markdown(html: str, url: str = "") -> str:
    """
    Convert HTML to clean markdown/text for LLM consumption.

    Strategy:
    1. trafilatura content extraction
    2. markdownify fallback
    3. simple tag stripping
    """
    if not html:
        return ""

    try:
        import trafilatura

        result = trafilatura.extract(
            html,
            url=url,
            include_links=True,
            include_tables=True,
            include_images=False,
            include_comments=False,
            output_format="txt",
            favor_recall=True,
        )
        if result and len(result) > 100:
            return _clean_text(result)
    except Exception as e:
        logger.debug(f"trafilatura failed: {e}")

    try:
        from markdownify import markdownify as md

        result = md(
            html,
            heading_style="ATX",
            bullets="-",
            strip=["script", "style", "nav", "footer", "header", "aside"],
        )
        if result and len(result) > 50:
            return _clean_text(result)
    except Exception as e:
        logger.debug(f"markdownify failed: {e}")

    return _strip_tags(html)


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def _strip_tags(html: str) -> str:
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    html = re.sub(r"<[^>]+>", " ", html)
    for entity, char in (
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
    ):
        html = html.replace(entity, char)
    return _clean_text(html)


class WebFetchTool(Tool):
    name = "fetch_url"
    description = (
        "Fetch a web page and convert it to readable markdown. "
        "Use this after search_web to read the full content of a promising result. "
        "If the page is too long, it will be truncated and cached for later access."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the web page to fetch.",
            },
        },
        "required": ["url"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(
        self,
        max_result_size_chars: int = 50000,
        http_timeout: int = 30,
        jina_timeout: int = 60,
        extraction_threshold: int = 15000,
    ):
        self.max_result_size_chars = max_result_size_chars
        self._extraction_threshold = extraction_threshold
        self._client = httpx.AsyncClient(
            timeout=float(http_timeout),
            follow_redirects=True,
            headers=DEFAULT_HEADERS,
        )
        self._jina_client = httpx.AsyncClient(timeout=float(jina_timeout))

    def prompt(self) -> str:
        return (
            "Use fetch_url to read the full content of a web page. Tips:\n"
            "- Always search_web first, then fetch the most relevant results\n"
            "- Don't fetch pages that are clearly irrelevant based on their title/snippet\n"
            "- Prefer pages from authoritative sources"
        )

    def validate_input(self, args: dict) -> ValidationResult:
        url = args.get("url", "")
        if not url:
            return ValidationResult(valid=False, message="URL is required")
        if not url.startswith(("http://", "https://")):
            return ValidationResult(
                valid=False,
                message="URL must start with http:// or https://",
            )
        is_safe, reason = validate_url_for_ssrf(url)
        if not is_safe:
            return ValidationResult(valid=False, message=f"URL blocked (SSRF protection): {reason}")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        url = args["url"]

        is_safe, reason = await validate_url_for_ssrf_async(url)
        if not is_safe:
            return ToolResult(
                data=f"URL blocked (SSRF protection): {reason}",
                is_error=True,
            )

        lower_url = url.lower()
        if any(marker in lower_url for marker in _SEARCH_ENGINE_URL_MARKERS):
            return ToolResult(
                data="Cannot fetch search engine result pages. Use search_web tool instead.",
                is_error=False,
            )

        if context.rate_limiter:
            await context.rate_limiter.acquire(url)

        jina_result = await self._fetch_via_jina(url, context)
        if jina_result is not None:
            return jina_result

        logger.info(f"Jina unavailable, falling back to direct fetch for {url}")
        return await self._fetch_direct(url, context)

    async def _fetch_via_jina(self, url: str, context: ToolUseContext) -> ToolResult | None:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Return-Format": "markdown",
            "X-No-Cache": "true",
            "X-Retain-Images": "none",
            "X-Remove-Selector": "nav, footer, .sidebar, .ads, .cookie-banner",
        }
        jina_key = os.environ.get("JINA_API_KEY", "")
        if jina_key:
            headers["Authorization"] = f"Bearer {jina_key}"

        try:
            response = await self._jina_client.get(f"{JINA_READER_URL}{url}", headers=headers)
            response.raise_for_status()
        except httpx.TimeoutException:
            logger.warning(f"Jina timeout for {url}")
            return None
        except httpx.HTTPStatusError as e:
            logger.warning(f"Jina HTTP error {e.response.status_code} for {url}")
            return None
        except Exception as e:
            logger.warning(f"Jina request failed for {url}: {e}")
            return None

        try:
            data = response.json()
            jina_data = data.get("data", {})
            markdown = jina_data.get("content", "")
            title = jina_data.get("title", "") or "Untitled Page"
        except Exception:
            markdown = response.text
            title = self._extract_title_from_markdown(markdown)

        if not markdown or len(markdown.strip()) < 50:
            logger.info(f"Jina returned insufficient content for {url}, falling back")
            return None

        full_content = f"## {title}\n**Source**: {url}\n\n{markdown}"
        result = await self._maybe_extract_or_truncate(
            full_content, url, title, context
        )
        citation = Citation(
            url=url,
            title=title,
            snippet=markdown[:300],
            source_type=SourceType.WEB,
        )
        if result is not None:
            result.citations.append(citation)
            return result

        return ToolResult(
            data=full_content,
            citations=[citation],
        )

    async def _fetch_direct(self, url: str, context: ToolUseContext) -> ToolResult:
        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx.TimeoutException:
            return ToolResult(
                data=f"Timeout fetching {url} — the page took too long to respond.",
                is_error=True,
            )
        except httpx.HTTPStatusError as e:
            return ToolResult(
                data=(
                    f"HTTP error fetching {url}: "
                    f"{e.response.status_code} {e.response.reason_phrase}"
                ),
                is_error=True,
            )
        except Exception as e:
            return ToolResult(data=f"Failed to fetch {url}: {str(e)}", is_error=True)

        content_type = response.headers.get("content-type", "")

        if "application/pdf" in content_type:
            return ToolResult(
                data=(
                    "This URL points to a PDF file. PDF parsing is not supported in direct mode. "
                    f"Set JINA_API_KEY to enable PDF reading via Jina. URL: {url}"
                ),
                is_error=False,
            )

        if "text/html" not in content_type and "application/xhtml" not in content_type:
            text = response.text[: self.max_result_size_chars]
            return ToolResult(
                data=f"## Content from {url}\n\n(Content-Type: {content_type})\n\n{text}",
                citations=[
                    Citation(
                        url=url,
                        title=url,
                        snippet=text[:200],
                        source_type=SourceType.WEB,
                    )
                ],
            )

        html = response.text
        markdown = _html_to_markdown(html, url=url)
        if not markdown or len(markdown) < 50:
            return ToolResult(
                data=(
                    f"Could not extract meaningful content from {url}. "
                    "The page may be JavaScript-heavy. "
                    "Set JINA_API_KEY to handle JS-rendered pages via Jina Reader."
                ),
                is_error=False,
            )

        title = self._extract_title(html, markdown)
        full_content = f"## {title}\n**Source**: {url}\n\n{markdown}"
        result = await self._maybe_extract_or_truncate(
            full_content, url, title, context
        )
        citation = Citation(
            url=url,
            title=title,
            snippet=markdown[:300],
            source_type=SourceType.WEB,
        )
        if result is not None:
            result.citations.append(citation)
            return result

        return ToolResult(
            data=full_content,
            citations=[citation],
        )

    async def _maybe_extract_or_truncate(
        self,
        full_content: str,
        url: str,
        title: str,
        context: ToolUseContext,
    ) -> ToolResult | None:
        """
        Handle large content: LLM extraction first, then raw truncation.

        Returns None if content is small enough to return raw.
        """
        if len(full_content) <= self._extraction_threshold:
            return None

        cached_path = await self._cache_full_content(full_content, url, context)

        if self._extraction_threshold > 0:
            from src.utils.content_extractor import extract_content

            research_query = context.extra.get("research_query", "")
            extracted = await extract_content(
                raw_content=full_content,
                research_query=research_query,
                source_url=url,
                source_title=title,
            )

            if extracted:
                extracted += (
                    f"\n\n---\n[Full content ({len(full_content):,} chars) "
                    f"cached at: {cached_path}. Use deep_read to access "
                    f"specific sections.]"
                )
                return ToolResult(
                    data=extracted,
                    citations=[],
                    truncated=True,
                    cached_path=str(cached_path),
                )

        preview = full_content[: self.max_result_size_chars]
        preview += (
            f"\n\n---\n[Content truncated. Full content "
            f"({len(full_content):,} chars) saved to: {cached_path}]"
        )
        return ToolResult(
            data=preview,
            citations=[],
            truncated=True,
            cached_path=str(cached_path),
        )

    async def _cache_full_content(
        self,
        content: str,
        url: str,
        context: ToolUseContext,
    ) -> str:
        import hashlib

        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cached_path = context.cache_dir / f"{self.name}_{url_hash}.md"
        cached_path.write_text(content, encoding="utf-8")
        logger.info(f"Cached {len(content):,} chars to {cached_path}")
        return str(cached_path)

    def _extract_title(self, html: str, markdown: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            title = re.split(r"\s*[|–—-]\s*(?=[A-Z])", title)[0].strip()
            if title:
                return title
        return self._extract_title_from_markdown(markdown)

    def _extract_title_from_markdown(self, markdown: str) -> str:
        match = re.match(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Untitled Page"
