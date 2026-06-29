"""Tests for RAGChain logic (memory, source formatting, answer flow).

RAGChain.__init__ loads heavy models and opens network clients, so we build
instances with __new__ and inject mocks for the pieces we exercise.
"""

from collections import defaultdict
from unittest.mock import MagicMock

import src.rag as rag_module
from src.rag import RAGChain


def make_rag() -> RAGChain:
    rag = RAGChain.__new__(RAGChain)
    rag._history = defaultdict(list)
    rag._llm = MagicMock()
    rag._embedder = MagicMock()
    rag._qdrant = MagicMock()
    return rag


def test_format_sources_dedupes_and_sorts_pages():
    rag = make_rag()
    docs = [
        {"page": 42, "text": "a", "score": 0.9},
        {"page": 10, "text": "b", "score": 0.8},
        {"page": 42, "text": "c", "score": 0.7},
    ]
    result = rag._format_sources(docs)
    assert result == "_Sources: p.10, p.42_"


def test_format_sources_empty():
    assert make_rag()._format_sources([]) == ""


def test_memory_window_caps_history():
    rag = make_rag()
    monkey_window = rag_module.MEMORY_WINDOW
    chat_id = 1
    for i in range(monkey_window + 5):
        rag._update_history(chat_id, f"q{i}", f"a{i}")
    # History holds at most MEMORY_WINDOW * 2 messages (user + assistant pairs).
    assert len(rag._history[chat_id]) == monkey_window * 2
    # Oldest entries are dropped; the most recent question survives.
    assert rag._history[chat_id][-2]["content"] == f"q{monkey_window + 4}"


def test_clear_history():
    rag = make_rag()
    rag._update_history(7, "q", "a")
    assert rag._history[7]
    rag.clear_history(7)
    assert rag._history[7] == []


def test_retrieve_uses_query_points_api():
    """Guards against the qdrant-client 1.18 break: search() was removed in
    favour of query_points(). This test fails if we regress to search()."""
    rag = make_rag()
    emb = MagicMock()
    emb.tolist.return_value = [0.1, 0.2, 0.3]
    rag._embedder.embed.return_value = iter([emb])

    point = MagicMock(score=0.91)
    point.payload = {"text": "chunk text", "page": 5}
    rag._qdrant.query_points.return_value = MagicMock(points=[point])

    docs = rag._retrieve("what is x?")

    rag._qdrant.query_points.assert_called_once()
    assert docs == [{"text": "chunk text", "page": 5, "score": 0.91}]


def test_answer_returns_message_when_no_context(monkeypatch):
    rag = make_rag()
    monkeypatch.setattr(rag, "_retrieve", lambda q: [])
    out = rag.answer(99, "something unrelated")
    assert "couldn't find" in out.lower()
    # The no-context exchange is still recorded in memory.
    assert len(rag._history[99]) == 2
    # The LLM is not called when there is no relevant context.
    rag._llm.invoke.assert_not_called()


def test_answer_happy_path_appends_sources(monkeypatch):
    rag = make_rag()
    monkeypatch.setattr(
        rag, "_retrieve", lambda q: [{"page": 5, "text": "ctx", "score": 0.9}]
    )
    rag._llm.invoke.return_value = MagicMock(content="Gradient descent minimizes loss.")

    out = rag.answer(1, "What is gradient descent?")

    assert "Gradient descent minimizes loss." in out
    assert "_Sources: p.5_" in out
    rag._llm.invoke.assert_called_once()
    # Question + answer stored in history.
    assert rag._history[1][0]["content"] == "What is gradient descent?"


def test_answer_includes_history_in_llm_call(monkeypatch):
    rag = make_rag()
    monkeypatch.setattr(
        rag, "_retrieve", lambda q: [{"page": 1, "text": "ctx", "score": 0.9}]
    )
    rag._llm.invoke.return_value = MagicMock(content="answer")

    rag.answer(1, "first question")
    rag.answer(1, "follow up")

    # Second call should include the first exchange as prior messages.
    second_call_messages = rag._llm.invoke.call_args.args[0]
    contents = [m.content for m in second_call_messages]
    assert any("first question" in c for c in contents)
