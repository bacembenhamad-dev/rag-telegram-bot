"""Tests for the Telegram handlers in src.bot (async, with mocked Update/Context)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest

import src.bot as bot


def make_message():
    """A message stub whose reply_text / chat.send_action are awaitable mocks."""
    chat = SimpleNamespace(id=1, send_action=AsyncMock())
    message = SimpleNamespace(chat=chat, reply_text=AsyncMock())
    return message


def make_context(rag):
    return SimpleNamespace(bot_data={"rag": rag})


def test_examples_keyboard_has_one_button_per_question():
    markup = bot._examples_keyboard()
    flat = [btn for row in markup.inline_keyboard for btn in row]
    assert len(flat) == len(bot.EXAMPLE_QUESTIONS)
    assert all(btn.callback_data.startswith("ask:") for btn in flat)


async def test_send_answer_falls_back_to_plain_text_on_bad_markdown():
    message = make_message()
    message.reply_text = AsyncMock(side_effect=[BadRequest("can't parse entities"), None])

    await bot._send_answer(message, "Broken *markdown")

    assert message.reply_text.await_count == 2
    # The retry must NOT pass a parse_mode (plain text).
    retry_kwargs = message.reply_text.await_args_list[1].kwargs
    assert "parse_mode" not in retry_kwargs


async def test_send_answer_splits_long_text():
    message = make_message()
    long_text = "word " * 2000  # > 4096 chars
    await bot._send_answer(message, long_text)
    assert message.reply_text.await_count > 1


async def test_handle_message_answers_and_replies():
    rag = MagicMock()
    rag.answer.return_value = "Here is your answer."
    message = make_message()
    update = SimpleNamespace(
        message=SimpleNamespace(text="What is a tensor?", chat=message.chat,
                                reply_text=message.reply_text),
        effective_chat=SimpleNamespace(id=1),
    )

    await bot.handle_message(update, make_context(rag))

    rag.answer.assert_called_once_with(1, "What is a tensor?")
    message.reply_text.assert_awaited()


async def test_handle_message_ignores_empty_text():
    rag = MagicMock()
    update = SimpleNamespace(
        message=SimpleNamespace(text="   ", chat=None, reply_text=AsyncMock()),
        effective_chat=SimpleNamespace(id=1),
    )
    await bot.handle_message(update, make_context(rag))
    rag.answer.assert_not_called()


async def test_answer_question_surfaces_friendly_error_on_failure():
    rag = MagicMock()
    rag.answer.side_effect = RuntimeError("LLM down")
    message = make_message()

    await bot._answer_question(make_context(rag), message, 1, "anything")

    message.reply_text.assert_awaited()
    sent = message.reply_text.await_args_list[0].args[0]
    assert "something went wrong" in sent.lower()


async def test_on_example_click_resolves_question():
    rag = MagicMock()
    rag.answer.return_value = "Answer text."
    chat = SimpleNamespace(id=1, send_action=AsyncMock())
    callback_message = SimpleNamespace(chat=chat, reply_text=AsyncMock())
    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            data="ask:1", answer=AsyncMock(), message=callback_message
        )
    )

    await bot.on_example_click(update, make_context(rag))

    update.callback_query.answer.assert_awaited()  # spinner dismissed
    rag.answer.assert_called_once_with(1, bot.EXAMPLE_QUESTIONS[1])


async def test_handle_non_text_replies_with_hint():
    update = SimpleNamespace(message=SimpleNamespace(reply_text=AsyncMock()))
    await bot.handle_non_text(update, make_context(MagicMock()))
    update.message.reply_text.assert_awaited()
    sent = update.message.reply_text.await_args.args[0]
    assert "text" in sent.lower()
