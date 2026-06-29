"""
Shared test setup. Runs before test modules are imported, so the dummy
credentials are in place when src.rag / src.bot read os.environ at import time.
"""

import os
import sys
from pathlib import Path

# Make the project root importable (so `import src.*` works from anywhere).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Dummy credentials — tests never make real network calls.
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:test-token")
os.environ.setdefault("COLLECTION_NAME", "ml_book")
os.environ.setdefault("MEMORY_WINDOW", "10")
