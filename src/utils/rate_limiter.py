"""
Per-domain rate limiting for web requests.

Uses aiolimiter for async-compatible rate limiting. Each domain gets
its own rate limiter to prevent overwhelming any single server while
allowing concurrent requests to different domains.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_AsyncLimiter = None
_limiter_resolved = False


def _get_limiter_class():
    global _AsyncLimiter, _limiter_resolved
    if not _limiter_resolved:
        try:
            from aiolimiter import AsyncLimiter

            _AsyncLimiter = AsyncLimiter
        except ImportError:
            _AsyncLimiter = None
        _limiter_resolved = True
    return _AsyncLimiter


class DomainRateLimiter:
    """Per-domain rate limiter for web requests."""

    def __init__(self, max_per_minute: int = 10):
        self.max_per_minute = max_per_minute
        self._limiters: dict[str, object] = {}

    def _get_domain(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            return parsed.netloc or url
        except Exception:
            return url

    def _get_limiter(self, domain: str):
        if domain not in self._limiters:
            limiter_class = _get_limiter_class()
            if limiter_class is not None:
                self._limiters[domain] = limiter_class(
                    max_rate=self.max_per_minute,
                    time_period=60,
                )
            else:
                self._limiters[domain] = None
        return self._limiters[domain]

    async def acquire(self, url: str) -> None:
        domain = self._get_domain(url)
        limiter = self._get_limiter(domain)

        if limiter is not None:
            await limiter.acquire()
            logger.debug(f"Rate limit acquired for {domain}")
        else:
            logger.debug(f"Rate limiting disabled (no aiolimiter), skipping for {domain}")
