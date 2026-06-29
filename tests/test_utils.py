"""Tests for src.utils.split_message."""

from src.utils import split_message


def test_short_text_is_single_chunk():
    text = "A short answer."
    assert split_message(text) == [text]


def test_text_at_limit_is_single_chunk():
    text = "x" * 4096
    assert split_message(text) == [text]


def test_long_text_splits_into_multiple_chunks():
    text = "word " * 2000  # ~10000 chars
    chunks = split_message(text)
    assert len(chunks) > 1
    assert all(len(c) <= 4096 for c in chunks)


def test_split_preserves_all_words():
    text = "\n\n".join(f"Paragraph number {i} with some content." for i in range(400))
    chunks = split_message(text)
    rejoined = " ".join(chunks)
    for i in range(400):
        assert f"Paragraph number {i}" in rejoined


def test_prefers_paragraph_boundary():
    first = "A" * 3000
    second = "B" * 3000
    chunks = split_message(first + "\n\n" + second)
    # The break should land on the blank line, keeping each block intact.
    assert chunks[0] == first
    assert chunks[1] == second


def test_hard_split_when_no_boundary():
    text = "z" * 9000  # no spaces or newlines at all
    chunks = split_message(text)
    assert all(len(c) <= 4096 for c in chunks)
    assert "".join(chunks) == text
