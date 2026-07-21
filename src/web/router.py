"""FastAPI routes — health check, REST API, and WebSocket research stream."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import yaml
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.context import ContextBuilder
from src.core.loop import QueryParams, query_loop
from src.core.tool import ToolRegistry, build_default_registry, register_skill_tools
from src.core.types import EventType, Message
from src.hooks.builtin_hooks import build_default_hooks
from src.hooks.engine import HookEngine
from src.hooks.plan_completeness_hook import PlanCompletenessHook
from src.llm.client import LLMClient, ModelConfig, set_shared_config
from src.memory.extract import extract_memories
from src.memory.retrieval import find_relevant_memories, format_memories_for_prompt
from src.memory.store import MemoryStore
from src.utils.docx_export import DOCX_MEDIA_TYPE, markdown_to_docx_bytes, safe_docx_filename
from src.utils.rate_limiter import DomainRateLimiter
from src.utils.session_storage import SessionStorage

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


def _server_api_key(settings: dict) -> str:
    return os.environ.get("SEARCH_CLAW_API_KEY") or settings.get("server", {}).get("api_key", "")


def _build_hook_engine(settings: dict) -> HookEngine:
    engine = HookEngine()
    for hook in build_default_hooks(settings.get("hooks", {})):
        engine.register_stop_hook(hook)
    engine.register_stop_hook(PlanCompletenessHook())
    return engine


def _build_app_state(
    settings: dict,
) -> tuple[
    ModelConfig,
    LLMClient,
    ToolRegistry,
    ContextBuilder,
    HookEngine,
    MemoryStore,
    SessionStorage,
    DomainRateLimiter,
    dict,
]:
    model_config = ModelConfig.from_settings("config/settings.yaml")
    set_shared_config(model_config)

    limits = settings.get("limits", {})
    memory_cfg = settings.get("memory", {})
    skills_cfg = settings.get("skills", {})
    sessions_cfg = settings.get("sessions", {})
    server_cfg = settings.get("server", {})
    tools_cfg = _tools_config(settings)
    cache_dir = tools_cfg.get("cache_dir", "./cache")

    tool_registry = build_default_registry(tools_cfg)
    skills_count = register_skill_tools(tool_registry, settings)
    llm_client = LLMClient(model_config)
    context_builder = ContextBuilder()
    hook_engine = _build_hook_engine(settings)
    memory_store = MemoryStore(base_dir=memory_cfg.get("base_dir", "./memory"))
    session_storage = SessionStorage(base_dir=sessions_cfg.get("base_dir", "./sessions"))
    rate_limiter = DomainRateLimiter(max_per_minute=int(limits.get("rate_limit_per_domain", 30)))

    runtime = {
        "max_turns": int(limits.get("max_turns", 40)),
        "max_search": int(limits.get("max_search", 30)),
        "max_fetch": int(limits.get("max_fetch", 30)),
        "cache_dir": cache_dir,
        "max_query_length": 10000,
        "compact_threshold_tokens": int(limits.get("compact_threshold_tokens", 80000)),
        "memory_enabled": bool(memory_cfg.get("enabled", False)),
        "max_relevant_memories": int(memory_cfg.get("max_relevant_memories", 5)),
        "skills_enabled": bool(skills_cfg.get("enabled", False)),
        "skills_count": skills_count,
        "api_key": _server_api_key(settings),
        "cors_origins": server_cfg.get("cors_origins", ""),
    }
    return (
        model_config,
        llm_client,
        tool_registry,
        context_builder,
        hook_engine,
        memory_store,
        session_storage,
        rate_limiter,
        runtime,
    )


async def _memory_content_for_query(
    query: str,
    runtime: dict,
    memory_store: MemoryStore,
) -> str | None:
    if not runtime["memory_enabled"]:
        return None
    try:
        relevant_memories = await find_relevant_memories(
            query,
            memory_store,
            max_memories=runtime["max_relevant_memories"],
        )
        return format_memories_for_prompt(relevant_memories)
    except Exception as e:
        logger.warning(f"Memory retrieval failed: {e}")
        return None


def _save_session_turn(
    session_storage: SessionStorage,
    session_id: str,
    *,
    query: str,
    final_answer: str,
    turn_count: int,
    citations: list,
    plan_findings: str = "",
    turns: list[dict] | None = None,
) -> None:
    try:
        session_storage.save_session(
            session_id,
            {
                "query": query,
                "turns": turns
                or [
                    {
                        "query": query,
                        "final_answer": final_answer,
                        "turn_count": turn_count,
                        "num_citations": len(citations),
                    }
                ],
                "final_answer": final_answer,
                "turn_count": turn_count,
                "num_citations": len(citations),
                "citations": citations,
                "plan_findings": plan_findings,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to save session: {e}")


def create_app(settings: dict | None = None) -> FastAPI:
    settings = settings or _load_settings()
    (
        model_config,
        llm_client,
        tool_registry,
        context_builder,
        hook_engine,
        memory_store,
        session_storage,
        rate_limiter,
        runtime,
    ) = _build_app_state(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("DeepsearchHarness starting")
        yield
        logger.info("Shutting down")
        await tool_registry.close_all()

    app = FastAPI(title="DeepsearchHarness", version="0.1.0", lifespan=lifespan)

    cors_origins = runtime["cors_origins"]
    if cors_origins:
        origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    api_key = runtime["api_key"]
    if api_key:

        class APIKeyAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                path = request.url.path
                if path in ("/api/health", "/api/login") or path.startswith("/static"):
                    return await call_next(request)
                if path == "/":
                    return await call_next(request)

                auth_header = request.headers.get("authorization", "")
                if auth_header.startswith("Bearer ") and auth_header[7:] == api_key:
                    return await call_next(request)
                if request.query_params.get("api_key") == api_key:
                    return await call_next(request)

                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key"},
                )

        app.add_middleware(APIKeyAuthMiddleware)

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
            "hooks": [h.name for h in hook_engine.stop_hooks],
            "memory_enabled": runtime["memory_enabled"],
            "skills_enabled": runtime["skills_enabled"],
            "skills_count": runtime["skills_count"],
        }

    @app.post("/api/login")
    async def login(request: Request):
        if not api_key:
            return {"ok": True}
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
        if body.get("password") == api_key:
            return {"ok": True}
        return JSONResponse(status_code=401, content={"error": "Wrong password"})

    @app.get("/api/sessions")
    async def list_sessions():
        return {"sessions": session_storage.list_sessions(limit=20)}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        session = session_storage.load_session(session_id)
        if session is None:
            return JSONResponse(content={"error": "Session not found"}, status_code=404)
        return session

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        if session_storage.delete_session(session_id):
            return {"ok": True}
        return JSONResponse(content={"error": "Session not found"}, status_code=404)

    @app.post("/api/export/docx")
    async def export_docx(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

        markdown = (body.get("markdown") or "").strip()
        if not markdown:
            return JSONResponse(status_code=400, content={"error": "markdown is required"})
        if len(markdown) > 500_000:
            return JSONResponse(status_code=400, content={"error": "Content too large to export."})

        title = body.get("title") or "research-report"
        try:
            docx_bytes = markdown_to_docx_bytes(markdown)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        except Exception as exc:
            logger.error("DOCX export failed: %s", exc, exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"DOCX export failed: {exc}"})

        filename = safe_docx_filename(title)
        return Response(
            content=docx_bytes,
            media_type=DOCX_MEDIA_TYPE,
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
        )

    @app.post("/api/query")
    async def api_query(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

        query = body.get("query", "").strip()
        if not query:
            return JSONResponse(status_code=400, content={"error": "Query is required"})
        if len(query) > runtime["max_query_length"]:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Query too long ({len(query)} chars). "
                        f"Maximum is {runtime['max_query_length']}."
                    ),
                },
            )

        session_id = str(uuid.uuid4())
        memory_content = await _memory_content_for_query(query, runtime, memory_store)
        system_prompt = context_builder.build_system_prompt(
            tool_registry.all_tools(),
            memory_content=memory_content,
        )
        params = QueryParams(
            query=query,
            system_prompt=system_prompt,
            tool_registry=tool_registry,
            llm_client=llm_client,
            max_turns=int(body.get("max_turns", runtime["max_turns"])),
            max_search=int(body.get("max_search", runtime["max_search"])),
            max_fetch=int(body.get("max_fetch", runtime["max_fetch"])),
            hook_engine=hook_engine,
            session_id=session_id,
            cache_dir=runtime["cache_dir"],
            compact_threshold_tokens=runtime["compact_threshold_tokens"],
            rate_limiter=rate_limiter,
        )

        answer = ""
        citations_list: list = []
        turn_count = 0
        plan_findings = ""
        gen = query_loop(params)
        sent_value: str | None = None

        try:
            while True:
                event = await gen.asend(sent_value)
                sent_value = None

                if event.type == EventType.USER_QUESTION:
                    options = event.data.get("options", [])
                    sent_value = options[0]["label"] if options else ""

                elif event.type == EventType.DONE:
                    answer = event.data.get("final_answer", "")
                    citations_list = event.data.get("citations", [])
                    turn_count = event.data.get("turn_count", 0)
                    plan_findings = event.data.get("plan_findings", "") or ""

        except StopAsyncIteration:
            pass
        except Exception as e:
            logger.error(f"API query error [{session_id[:8]}]: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"Research failed: {str(e)}"})

        _save_session_turn(
            session_storage,
            session_id,
            query=query,
            final_answer=answer,
            turn_count=turn_count,
            citations=citations_list,
            plan_findings=plan_findings,
        )

        if runtime["memory_enabled"] and answer:
            asyncio.create_task(
                extract_memories(
                    query=query,
                    final_answer=answer,
                    plan_findings=plan_findings,
                    store=memory_store,
                )
            )

        return {
            "answer": answer,
            "citations": citations_list,
            "turn_count": turn_count,
            "session_id": session_id,
        }

    @app.websocket("/ws/search")
    async def search_websocket(ws: WebSocket):
        if api_key:
            api_key_param = ws.query_params.get("api_key", "")
            if api_key_param != api_key:
                await ws.close(code=4001, reason="Invalid or missing API key")
                return

        await ws.accept()
        logger.info("WebSocket connected")

        conversation_history: list[Message] = []
        session_turns: list[dict] = []
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
                    await ws.send_json(
                        {
                            "type": "error",
                            "data": {
                                "message": (
                                    f"Query too long ({len(query)} chars). "
                                    f"Maximum is {runtime['max_query_length']}."
                                ),
                            },
                        }
                    )
                    continue

                if message.get("new_chat"):
                    conversation_history.clear()
                    session_turns.clear()
                    session_id = str(uuid.uuid4())

                memory_content = await _memory_content_for_query(query, runtime, memory_store)
                system_prompt = context_builder.build_system_prompt(
                    tool_registry.all_tools(),
                    memory_content=memory_content,
                )
                params = QueryParams(
                    query=query,
                    system_prompt=system_prompt,
                    tool_registry=tool_registry,
                    llm_client=llm_client,
                    history=list(conversation_history),
                    max_turns=runtime["max_turns"],
                    max_search=runtime["max_search"],
                    max_fetch=runtime["max_fetch"],
                    hook_engine=hook_engine,
                    session_id=session_id,
                    cache_dir=runtime["cache_dir"],
                    compact_threshold_tokens=runtime["compact_threshold_tokens"],
                    rate_limiter=rate_limiter,
                )

                final_answer = ""
                plan_findings = ""
                citations_list: list = []
                turn_count = 0
                try:
                    async for event in query_loop(params):
                        if event.type == EventType.DONE:
                            event.data["session_id"] = session_id
                            final_answer = event.data.get("final_answer", "") or ""
                            plan_findings = event.data.get("plan_findings", "") or ""
                            citations_list = event.data.get("citations", [])
                            turn_count = event.data.get("turn_count", 0)
                        await ws.send_json(event.to_dict())
                except Exception as e:
                    logger.error(f"Query loop error: {e}", exc_info=True)
                    await ws.send_json({"type": "error", "data": {"message": str(e)}})
                    continue

                if final_answer:
                    session_turns.append(
                        {
                            "query": query,
                            "final_answer": final_answer,
                            "turn_count": turn_count,
                            "num_citations": len(citations_list),
                        }
                    )
                    _save_session_turn(
                        session_storage,
                        session_id,
                        query=query,
                        final_answer=final_answer,
                        turn_count=turn_count,
                        citations=citations_list,
                        plan_findings=plan_findings,
                        turns=session_turns,
                    )

                if runtime["memory_enabled"] and final_answer:
                    asyncio.create_task(
                        extract_memories(
                            query=query,
                            final_answer=final_answer,
                            plan_findings=plan_findings,
                            store=memory_store,
                        )
                    )

                conversation_history.append(Message(role="user", content=query))
                if final_answer:
                    conversation_history.append(Message(role="assistant", content=final_answer))

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)

    return app
