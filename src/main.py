"""
DeepsearchHarness — FastAPI entry point.

Usage:
    python -m src.main
"""

from __future__ import annotations

import logging
import os

import uvicorn

from src.web.router import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
app = create_app()


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    try:
        import yaml

        with open("config/settings.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        server = cfg.get("server", {})
        host = server.get("host", host)
        port = int(server.get("port", port))
    except Exception:
        pass

    host = os.environ.get("HOST", host)
    port = int(os.environ.get("PORT", str(port)))

    llm_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not llm_key:
        logger.warning("No OPENAI_API_KEY / DASHSCOPE_API_KEY — LLM calls will fail until set.")

    logger.info(f"Starting DeepsearchHarness on http://{host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
