"""Tests for local skill loader."""

from __future__ import annotations

from pathlib import Path

from src.skills.loader import load_skills


def _write_skill(root: Path, name: str, body: str, frontmatter: str = "") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    if frontmatter:
        content = f"---\n{frontmatter}\n---\n\n{body}"
    else:
        content = body
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def test_load_skills_parses_frontmatter(tmp_path: Path):
    _write_skill(
        tmp_path,
        "demo-skill",
        "# Demo\nUse $ARGUMENTS here.",
        "name: demo\ndescription: Demo skill\nwhen_to_use: For tests",
    )

    skills = load_skills([tmp_path], root=tmp_path)

    assert len(skills) == 1
    assert skills[0].name == "demo"
    assert skills[0].description == "Demo skill"
    assert "$ARGUMENTS" in skills[0].content


def test_later_skill_dir_overrides_name(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_skill(first, "shared", "First body", "name: shared\ndescription: first")
    _write_skill(second, "shared", "Second body", "name: shared\ndescription: second")

    skills = load_skills([first, second], root=tmp_path)
    by_name = {skill.name: skill for skill in skills}

    assert by_name["shared"].content.startswith("Second")
