"""Tests for per-domain rate limiting."""

from __future__ import annotations

import pytest

from src.utils.rate_limiter import DomainRateLimiter


@pytest.mark.asyncio
async def test_acquire_does_not_raise_for_url():
    limiter = DomainRateLimiter(max_per_minute=100)
    await limiter.acquire("https://example.com/page")
    await limiter.acquire("https://other.example.org/article")
