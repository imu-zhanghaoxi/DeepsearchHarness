"""Tests for use_skill and run_skill_script tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.tool import ToolUseContext
from src.skills.loader import Skill, load_skills
from src.tools.run_skill_script import RunSkillScriptTool
from src.tools.use_skill import UseSkillTool


@pytest.fixture
def skill_root(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\nTopic: $ARGUMENTS\n",
        encoding="utf-8",
    )
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "hello.py").write_text(
        "import sys\nprint('hello ' + ' '.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    return tmp_path / "skills"


@pytest.fixture
def skills(skill_root: Path) -> list[Skill]:
    return load_skills([skill_root], root=skill_root.parent)


@pytest.fixture
def context(tmp_path) -> ToolUseContext:
    return ToolUseContext(cache_dir=tmp_path / "cache")


@pytest.mark.asyncio
async def test_use_skill_loads_content(skills: list[Skill], context: ToolUseContext):
    tool = UseSkillTool(skills=skills)

    result = await tool.call({"skill": "demo", "args": "python asyncio"}, context)

    assert result.is_error is False
    assert "Topic: python asyncio" in result.data
    assert "demo" in tool.prompt()


@pytest.mark.asyncio
async def test_use_skill_rejects_unknown_skill(skills: list[Skill], context: ToolUseContext):
    tool = UseSkillTool(skills=skills)

    result = await tool.call({"skill": "missing"}, context)

    assert result.is_error is True
    assert "Unknown skill" in result.data


@pytest.mark.asyncio
async def test_run_skill_script_executes_python(skills: list[Skill], context: ToolUseContext):
    tool = RunSkillScriptTool(skills=skills, default_timeout_seconds=5)

    result = await tool.call(
        {
            "skill": "demo",
            "script": "scripts/hello.py",
            "args": ["world"],
        },
        context,
    )

    assert result.is_error is False
    assert "hello world" in result.data


@pytest.mark.asyncio
async def test_run_skill_script_rejects_path_traversal(
    skills: list[Skill], context: ToolUseContext
):
    tool = RunSkillScriptTool(skills=skills)

    result = await tool.call(
        {
            "skill": "demo",
            "script": "../hello.py",
        },
        context,
    )

    assert result.is_error is True
    assert ".." in result.data or "relative" in result.data.lower()
