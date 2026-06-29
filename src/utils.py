"""
Pure helper functions with no Telegram/network dependencies — easy to unit-test.
"""

# Telegram rejects any single message longer than 4096 characters.
TELEGRAM_MAX_LEN = 4096


def split_message(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    """Split ``text`` into chunks no longer than ``limit`` characters.

    Breaks are preferred at paragraph boundaries, then line breaks, then
    spaces, so words and sentences stay intact across chunks.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > limit:
        window = remaining[:limit]
        split_at = window.rfind("\n\n")
        if split_at <= 0:
            split_at = window.rfind("\n")
        if split_at <= 0:
            split_at = window.rfind(" ")
        if split_at <= 0:
            split_at = limit  # no natural boundary — hard split

        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks
