"""Tests for SSRF URL validation."""

from src.utils.url_validator import validate_url_for_ssrf


class TestValidateUrlForSsrf:
    def test_blocks_localhost_ip(self):
        is_safe, reason = validate_url_for_ssrf("http://127.0.0.1/admin")
        assert is_safe is False
        assert "127.0.0.1" in reason

    def test_blocks_private_ip_literal(self):
        is_safe, reason = validate_url_for_ssrf("https://192.168.1.1/status")
        assert is_safe is False
        assert "private" in reason.lower() or "192.168" in reason

    def test_blocks_metadata_hostname(self):
        is_safe, reason = validate_url_for_ssrf("http://metadata.google.internal/")
        assert is_safe is False
        assert "metadata" in reason.lower()

    def test_blocks_non_http_scheme(self):
        is_safe, reason = validate_url_for_ssrf("file:///etc/passwd")
        assert is_safe is False
        assert "scheme" in reason.lower()

    def test_allows_public_https_url(self):
        is_safe, reason = validate_url_for_ssrf("https://example.com/article")
        assert is_safe is True
        assert reason == ""

    def test_requires_hostname(self):
        is_safe, reason = validate_url_for_ssrf("https:///missing-host")
        assert is_safe is False
