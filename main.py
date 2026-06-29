"""
Entry point.

Production (Render):  python main.py
Local dev (polling):  python main.py --local
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_polling() -> None:
    """Local development mode — no public URL needed."""
    from src.bot import build_application
    from src.rag import RAGChain

    rag = RAGChain()
    app = build_application(rag)

    logger.info("Starting bot in POLLING mode (local dev)...")
    app.run_polling(drop_pending_updates=True)


def run_webhook() -> None:
    """Production mode — Render provides the HTTPS URL."""
    import uvicorn
    from fastapi import FastAPI, Request, Response
    from src.bot import build_application
    from src.rag import RAGChain

    webhook_url = os.environ["WEBHOOK_URL"]
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    port = int(os.getenv("PORT", "10000"))

    rag = RAGChain()
    app = build_application(rag)

    fastapi_app = FastAPI()
    webhook_path = f"/webhook/{token}"

    @fastapi_app.on_event("startup")
    async def on_startup():
        await app.initialize()
        full_url = f"{webhook_url.rstrip('/')}{webhook_path}"
        await app.bot.set_webhook(url=full_url)
        logger.info("Webhook set to: %s", full_url)
        await app.start()

    @fastapi_app.on_event("shutdown")
    async def on_shutdown():
        await app.stop()
        await app.shutdown()

    @fastapi_app.post(webhook_path)
    async def telegram_webhook(request: Request) -> Response:
        data = await request.json()
        from telegram import Update
        update = Update.de_json(data, app.bot)
        await app.process_update(update)
        return Response(content="ok")

    @fastapi_app.get("/health")
    async def health():
        return {"status": "ok"}

    logger.info("Starting bot in WEBHOOK mode on port %d...", port)
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run in polling mode for local dev")
    args = parser.parse_args()

    if args.local:
        run_polling()
    else:
        run_webhook()
