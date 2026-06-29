"""
Telegram bot handlers. Supports both webhook (production) and polling (local dev).
"""

import asyncio
import logging
import os

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.rag import RAGChain
from src.utils import split_message

logger = logging.getLogger(__name__)

# How often to re-send the "typing…" action while the LLM is working.
# Telegram clears the indicator after ~5s, so we refresh it a bit faster.
TYPING_REFRESH_SECONDS = 4

WELCOME_MESSAGE = (
    "👋 *Welcome to the Hands-On ML Book Bot!*\n\n"
    "I answer questions about _Hands-On Machine Learning with Scikit-Learn, "
    "Keras, and TensorFlow_ by Aurélien Géron — straight from the book, with page citations.\n\n"
    "Just send me any question about machine learning concepts, code, or theory.\n\n"
    "Not sure where to start? Tap one of the example questions below 👇"
)

HELP_MESSAGE = (
    "*How to use this bot*\n\n"
    "• Ask specific questions for the best answers\n"
    "  _Example: \"How does gradient descent work?\"_\n\n"
    "• Reference chapters or topics directly\n"
    "  _Example: \"Explain SVMs from Chapter 5\"_\n\n"
    "• Follow up naturally — I remember our recent conversation\n"
    "  _Example: \"Can you give a code example of that?\"_\n\n"
    "*Commands*\n"
    "/start — welcome message & example questions\n"
    "/help — this message\n"
    "/about — what powers this bot\n"
    "/clear — reset our conversation memory"
)

ABOUT_MESSAGE = (
    "*About this bot* 🤖\n\n"
    "A Retrieval-Augmented Generation (RAG) assistant for the book "
    "_Hands-On Machine Learning_ (Aurélien Géron, 2019).\n\n"
    "Every answer is grounded in the actual book text — I search the most "
    "relevant passages first, then answer from them and cite the page numbers.\n\n"
    "*Under the hood*\n"
    "• Embeddings: sentence-transformers (MiniLM)\n"
    "• Vector search: Qdrant\n"
    "• LLM: Llama 3.3 70B via Groq\n\n"
    "_Built as a portfolio project._"
)

# Used both for the /start inline keyboard and for resolving button taps.
# callback_data is limited to 64 bytes, so buttons reference questions by index.
EXAMPLE_QUESTIONS = [
    "What is the difference between supervised and unsupervised learning?",
    "How does gradient descent work?",
    "What is regularization and why is it useful?",
    "Explain the bias/variance trade-off.",
]


def _examples_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(q, callback_data=f"ask:{i}")]
        for i, q in enumerate(EXAMPLE_QUESTIONS)
    ]
    return InlineKeyboardMarkup(buttons)


async def _send_answer(message, text: str) -> None:
    """Send ``text`` to the chat, splitting long answers and falling back to
    plain text if the LLM produced Markdown that Telegram can't parse."""
    for chunk in split_message(text):
        try:
            await message.reply_text(
                chunk, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True
            )
        except BadRequest:
            await message.reply_text(chunk, disable_web_page_preview=True)


async def _keep_typing(chat, stop_event: asyncio.Event) -> None:
    """Keep the 'typing…' indicator alive until ``stop_event`` is set."""
    try:
        while not stop_event.is_set():
            await chat.send_action(ChatAction.TYPING)
            await asyncio.sleep(TYPING_REFRESH_SECONDS)
    except asyncio.CancelledError:
        pass


async def _answer_question(
    context: ContextTypes.DEFAULT_TYPE, message, chat_id: int, question: str
) -> None:
    """Run retrieval + LLM (off the event loop) while showing a typing indicator."""
    rag: RAGChain = context.bot_data["rag"]

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(message.chat, stop_event))
    try:
        # rag.answer is blocking (embeddings + HTTP LLM call) — keep the
        # event loop free so the typing indicator keeps refreshing.
        response = await asyncio.to_thread(rag.answer, chat_id, question)
    except Exception:
        logger.exception("Error generating answer for chat_id=%s", chat_id)
        response = (
            "⚠️ Sorry, something went wrong while generating your answer. "
            "Please try again in a moment."
        )
    finally:
        stop_event.set()
        typing_task.cancel()

    await _send_answer(message, response)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        WELCOME_MESSAGE,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_examples_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.MARKDOWN)


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(ABOUT_MESSAGE, parse_mode=ParseMode.MARKDOWN)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rag: RAGChain = context.bot_data["rag"]
    rag.clear_history(update.effective_chat.id)
    await update.message.reply_text(
        "🧹 Conversation memory cleared. Ask me anything to start fresh!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = (update.message.text or "").strip()
    if not question:
        return
    await _answer_question(context, update.message, update.effective_chat.id, question)


async def on_example_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()  # stop the button's loading spinner

    try:
        index = int(query.data.split(":", 1)[1])
        question = EXAMPLE_QUESTIONS[index]
    except (IndexError, ValueError):
        return

    # Echo the chosen question so the chat reads naturally.
    await query.message.reply_text(f"❓ _{question}_", parse_mode=ParseMode.MARKDOWN)
    await _answer_question(context, query.message, query.message.chat.id, question)


async def handle_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "I can only answer *text* questions about the book. "
        "Please type your question 🙂",
        parse_mode=ParseMode.MARKDOWN,
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception: %s", context.error, exc_info=context.error)


async def _post_init(app: Application) -> None:
    """Register the slash-command menu shown in the Telegram UI."""
    await app.bot.set_my_commands(
        [
            BotCommand("start", "Welcome & example questions"),
            BotCommand("help", "How to use the bot"),
            BotCommand("about", "What powers this bot"),
            BotCommand("clear", "Reset conversation memory"),
        ]
    )


def build_application(rag: RAGChain) -> Application:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).post_init(_post_init).build()
    app.bot_data["rag"] = rag

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CallbackQueryHandler(on_example_click, pattern=r"^ask:\d+$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_non_text))
    app.add_error_handler(error_handler)

    return app
