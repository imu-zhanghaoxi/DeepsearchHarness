"""FastAPI routes — health check and WebSocket research stream."""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.core.context import ContextBuilder
from src.core.loop import QueryParams, query_loop
from src.core.tool import ToolRegistry, build_default_registry
from src.core.types import EventType, Message
from src.llm.client import LLMClient, ModelConfig, set_shared_config

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _load_settings(path: str = "config/settings.yaml") -> dict:
    settings_path = Path(path)
    if not settings_path.exists():
        return {}
    try:
        with open(settings_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load settings: {e}")
        return {}


def _tools_config(settings: dict) -> dict:
    tools = dict(settings.get("tools", {}))
    if os.environ.get("SEARXNG_URL"):
        tools["searxng_url"] = os.environ["SEARXNG_URL"]
    return tools


def _build_app_state(settings: dict) -> tuple[ModelConfig, LLMClient, ToolRegistry, ContextBuilder, dict]:
    model_config = ModelConfig.from_settings("config/settings.yaml")
    set_shared_config(model_config)

    limits = settings.get("limits", {})
    tools_cfg = _tools_config(settings)
    cache_dir = tools_cfg.get("cache_dir", "./cache")

    tool_registry = build_default_registry(tools_cfg)
    llm_client = LLMClient(model_config)
    context_builder = ContextBuilder()

    runtime = {
        "max_turns": int(limits.get("max_turns", 40)),
        "max_search": int(limits.get("max_search", 30)),
        "max_fetch": int(limits.get("max_fetch", 30)),
        "cache_dir": cache_dir,
        "max_query_length": 10000,
    }
    return model_config, llm_client, tool_registry, context_builder, runtime


def create_app(settings: dict | None = None) -> FastAPI:
    settings = settings or _load_settings()
    model_config, llm_client, tool_registry, context_builder, runtime = _build_app_state(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("DeepsearchHarness starting")
        yield
        logger.info("Shutting down")
        await tool_registry.close_all()

    app = FastAPI(title="DeepsearchHarness", version="0.1.0", lifespan=lifespan)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>DeepsearchHarness</h1><p>Static files not found.</p>")

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "model": model_config.default_model,
            "tools": [t.name for t in tool_registry.all_tools()],
        }

    @app.websocket("/ws/search")
    async def search_websocket(ws: WebSocket):
        await ws.accept()
        logger.info("WebSocket connected")

        conversation_history: list[Message] = []
        session_id = str(uuid.uuid4())

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "data": {"message": "Invalid JSON"}})
                    continue

                query = str(message.get("query", "")).strip()
                if not query:
                    await ws.send_json({"type": "error", "data": {"message": "Query is required"}})
                    continue

                if len(query) > runtime["max_query_length"]:
                    await ws.send_json({
                        "type": "error",
                        "data": {
                            "message": (
                                f"Query too long ({len(query)} chars). "
                                f"Maximum is {runtime['max_query_length']}."
                            ),
                        },
                    })
                    continue

                if message.get("new_chat"):
                    conversation_history.clear()
                    session_id = str(uuid.uuid4())

                system_prompt = context_builder.build_system_prompt(tool_registry.all_tools())
                params = QueryParams(
                    query=query,
                    system_prompt=system_prompt,
                    tool_registry=tool_registry,
                    llm_client=llm_client,
                    history=list(conversation_history),
                    max_turns=runtime["max_turns"],
                    max_search=runtime["max_search"],
                    max_fetch=runtime["max_fetch"],
                    session_id=session_id,
                    cache_dir=runtime["cache_dir"],
                )

                final_answer = ""
                try:
                    async for event in query_loop(params):
                        if event.type == EventType.DONE:
                            event.data["session_id"] = session_id
                            final_answer = event.data.get("final_answer", "") or ""
                        await ws.send_json(event.to_dict())
                except Exception as e:
                    logger.error(f"Query loop error: {e}", exc_info=True)
                    await ws.send_json({"type": "error", "data": {"message": str(e)}})
                    continue

                conversation_history.append(Message(role="user", content=query))
                if final_answer:
                    conversation_history.append(Message(role="assistant", content=final_answer))

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)

    return app
