"""Tests for DOCX export helpers (no pandoc required)."""

from src.utils.docx_export import (
    normalize_markdown_blockquotes,
    normalize_markdown_for_docx,
    normalize_markdown_tables,
    safe_docx_filename,
)


def test_safe_docx_filename_sanitizes():
    assert safe_docx_filename('report: "Q1"').endswith(".docx")
    assert "/" not in safe_docx_filename("a/b")


def test_normalize_blockquotes():
    source = "> quote one\n> quote two\n\nNormal paragraph"
    result = normalize_markdown_blockquotes(source)
    assert ">" not in result.split("\n")[0]
    assert "quote one" in result


def test_normalize_tables_inserts_separator():
    source = "| A | B |\n| 1 | 2 |"
    result = normalize_markdown_tables(source)
    assert "| --- | --- |" in result


def test_normalize_markdown_for_docx_combines_steps():
    source = "> summary\n\n| H1 | H2 |\n| v1 | v2 |"
    result = normalize_markdown_for_docx(source)
    assert "summary" in result
    assert "| --- |" in result
