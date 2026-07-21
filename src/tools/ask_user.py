"""
Interactive user question tool.

Allows the agent to pause mid-research and ask the user a clarifying
question with selectable options. The loop detects ``pending_question``
in result metadata, yields USER_QUESTION, and injects the answer.
"""

from __future__ import annotations

import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class AskUserTool(Tool):
    name = "ask_user"
    description = (
        "Ask the user a clarifying question when their query is ambiguous "
        "or could benefit from narrowing the scope. Present 2-5 clear, "
        "mutually exclusive options for the user to choose from."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "A clear, concise question to ask the user. Should explain "
                    "why you're asking and what difference the answer makes."
                ),
            },
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Short label for the option (1-5 words).",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief explanation of what this option means.",
                        },
                    },
                    "required": ["label"],
                },
                "minItems": 2,
                "maxItems": 5,
                "description": "Selectable options for the user to choose from.",
            },
        },
        "required": ["question", "options"],
    }

    is_concurrency_safe = False
    is_read_only = True
    max_result_size_chars = 1000

    def prompt(self) -> str:
        return (
            "Use ask_user to ask the user a clarifying question when their "
            "query is ambiguous or could go in multiple directions.\n\n"
            "Guidelines:\n"
            "- Only ask when clarification genuinely improves the research\n"
            "- Present 2-5 clear, distinct options\n"
            "- Do NOT include catch-all options like 'Other' — the UI provides free text\n"
            "- Maximum 1 question per research session"
        )

    def validate_input(self, args: dict) -> ValidationResult:
        question = args.get("question", "").strip()
        if not question:
            return ValidationResult(valid=False, message="Question is required")

        options = args.get("options", [])
        if not isinstance(options, list) or len(options) < 2:
            return ValidationResult(valid=False, message="At least 2 options are required")
        if len(options) > 5:
            return ValidationResult(valid=False, message="Maximum 5 options allowed")

        for i, opt in enumerate(options):
            if not isinstance(opt, dict) or not opt.get("label", "").strip():
                return ValidationResult(
                    valid=False,
                    message=f"Option {i + 1} must have a non-empty label",
                )

        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        question = args["question"]
        options = args["options"]

        logger.info(
            f'AskUser: "{question}" with {len(options)} options: '
            + ", ".join(o.get("label", "?") for o in options)
        )

        return ToolResult(
            data="",
            metadata={
                "pending_question": {
                    "question": question,
                    "options": options,
                }
            },
        )
