<div align="center">

# 🤖 Hands-On ML — Telegram RAG Bot

**Ask a Telegram bot anything about _Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow_ — and get answers grounded in the book, with page citations.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![python-telegram-bot](https://img.shields.io/badge/python--telegram--bot-21-26A5E4?logo=telegram&logoColor=white)](https://python-telegram-bot.org/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-DC244C)](https://qdrant.tech/)
[![Groq](https://img.shields.io/badge/LLM-Llama%203.3%2070B%20via%20Groq-F55036)](https://groq.com/)
[![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)](#-running-the-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 Overview

This is an end-to-end **Retrieval-Augmented Generation (RAG)** chatbot. Instead of letting a language model answer machine-learning questions from memory (and risk hallucinating), the bot:

1. **Retrieves** the most relevant passages from the actual book using semantic search, then
2. **Generates** an answer grounded *only* in those passages — and cites the page numbers.

The result is a friendly study assistant that never makes things up and always shows its sources. It runs entirely on **free tiers** (Groq + Qdrant Cloud + Render) and uses **local embeddings**, so there is no embedding-API cost.

> 💡 Although it ships configured for the Géron ML book, the pipeline is book-agnostic — point it at any text PDF.

---

## ✨ Features

| | |
|---|---|
| 📚 **Grounded answers** | Every reply is generated from the book's text and cites the source page numbers — no hallucinations. |
| 💬 **Conversation memory** | Remembers your recent messages so follow-ups like *"give me a code example of that"* just work. |
| 🔘 **One-tap examples** | `/start` shows tappable example questions so new users can try it in one click. |
| 📋 **Native command menu** | Slash commands register with Telegram and appear in the UI automatically. |
| ⌨️ **Live typing indicator** | Stays active while the model thinks, even on slow responses. |
| 🪶 **Robust delivery** | Auto-splits replies over Telegram's 4096-char limit and never crashes on malformed Markdown. |
| 🆓 **100% free stack** | Local embeddings + free LLM/vector/hosting tiers = zero running cost. |
| ✅ **Tested** | 21 unit tests covering retrieval logic, memory, formatting, and every bot handler. |

---

## 🏗️ Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │            INGESTION (one-time)               │
                          │                                              │
   📕 Book PDF  ─────────▶│  PyMuPDF → chunk (1000/200) → MiniLM embed   │
                          │                          │                   │
                          └──────────────────────────┼───────────────────┘
                                                     ▼
                                          ┌────────────────────┐
                                          │   Qdrant Cloud      │
                                          │  (vector database)  │
                                          └────────────────────┘
                                                     ▲
                          ┌──────────────────────────┼───────────────────┐
                          │            QUERY (per message)                │
   👤 User ──▶ Telegram ─▶│  embed question → top-5 search ──────────────┘
                          │        │                                      │
                          │        ▼                                      │
                          │  build prompt (system + history + context)    │
                          │        │                                      │
                          │        ▼                                      │
                          │  Groq · Llama 3.3 70B ──▶ answer + page cites │
   👤 User ◀── Telegram ◀─│                                              │
                          └──────────────────────────────────────────────┘
```

---

## 🧰 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Groq — `llama-3.3-70b-versatile` | Fast inference, generous free tier, open-weight model |
| **Embeddings** | fastembed — `all-MiniLM-L6-v2` (ONNX) | Runs locally, free, no PyTorch — ~10x less RAM, fits Render's 512 MB free tier (384-dim) |
| **Vector store** | Qdrant Cloud | Persistent, always-on, free 1 GB cluster |
| **PDF parsing** | PyMuPDF (`fitz`) | Reliable text extraction from complex technical PDFs |
| **RAG / orchestration** | LangChain | Glues retriever, prompt, and LLM together |
| **Bot framework** | python-telegram-bot v21 (async) | Webhook + polling support |
| **Web server** | FastAPI + Uvicorn | Receives Telegram webhooks in production |
| **Deployment** | Render | Free tier, auto HTTPS, git-push deploys |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- A text PDF to index (this project targets the Géron ML book)
- Three free accounts: **Telegram**, **Groq**, **Qdrant Cloud**

### 1. Clone & install

```bash
git clone https://github.com/bacembenhamad-dev/rag-telegram-bot.git
cd rag-telegram-bot
pip install -r requirements.txt
```

### 2. Get your credentials (all free)

| Credential | Where to get it |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Message [@BotFather](https://t.me/BotFather) → `/newbot` |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → **API Keys** |
| `QDRANT_URL` + `QDRANT_API_KEY` | [cloud.qdrant.io](https://cloud.qdrant.io) → create a cluster → **API Keys** |

### 3. Configure environment

```bash
cp .env.example .env
# open .env and fill in your values
```

### 4. Add the book & ingest it (one-time)

Drop your PDF into the `PDFs/` folder, then build the vector index:

```bash
python -m src.ingest
```

This extracts the text, splits it into overlapping chunks, embeds them locally, and uploads the vectors to Qdrant. Takes a few minutes. Expected output:

```
Done! Collection 'ml_book' now has 1343 vectors.
```

### 5. Run it locally

```bash
python main.py --local      # polling mode — no public URL needed
```

Open Telegram, find your bot, send `/start`, and start asking questions. 🎉

---

## 💬 Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message + tappable example questions |
| `/help` | Usage tips |
| `/about` | What powers the bot |
| `/clear` | Reset your conversation memory |

Or just send any question in plain text.

---

## 🧪 Running the Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite (**21 tests**) covers retrieval/memory logic, source formatting, message splitting, the Markdown-safe send fallback, and all Telegram handlers. Everything is mocked — **no API keys or network access required**.

---

## ☁️ Deploy to Render (free)

1. Push this repository to GitHub.
2. On [render.com](https://render.com) → **New → Blueprint** → connect this repo. Render reads `render.yaml` and configures the service for you.
3. When prompted, fill in the four secrets:
   `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `QDRANT_URL`, `QDRANT_API_KEY`.
4. Click **Apply** and wait for the build.

That's it — no second deploy needed. The bot derives its public URL from Render's `RENDER_EXTERNAL_URL` automatically and registers the Telegram webhook on startup. A `/health` endpoint is exposed for uptime checks.

> ⏳ Render's free tier sleeps after inactivity, so the first message after idle may take a few seconds to wake the service.

---

## 📂 Project Structure

```
rag-telegram-bot/
├── PDFs/                 ← place the book PDF here (git-ignored)
├── src/
│   ├── ingest.py         ← one-time PDF → chunks → embeddings → Qdrant
│   ├── rag.py            ← retrieval + Groq LLM + per-user memory
│   ├── bot.py            ← Telegram command & message handlers
│   └── utils.py          ← pure helpers (message splitting)
├── tests/                ← pytest suite (no network/keys needed)
├── main.py               ← entry point (--local = polling, default = webhook)
├── render.yaml           ← Render deployment config
├── pytest.ini            ← test configuration
├── requirements.txt      ← runtime dependencies
├── requirements-dev.txt  ← + test dependencies
├── .env.example          ← configuration template
└── README.md
```

---

## 🔍 How It Works

```
User message → Telegram → webhook → bot.py
    → RAGChain.answer(chat_id, question)
        → embed question         (fastembed / MiniLM, local)
        → search Qdrant          (top-5 relevant passages, score-filtered)
        → build prompt           (system + recent history + book context)
        → Groq · Llama 3.3 70B   (temperature 0.2, grounded answer)
        → append page citations
    → reply on Telegram  (auto-split if long, Markdown-safe)
```

The system prompt constrains the model to answer **only** from the retrieved context; if nothing relevant is found, it says so instead of guessing.

---

## 📜 License

Released under the [MIT License](LICENSE).

The book PDF is **not** included and is **not** covered by this license — supply your own legally obtained copy.

---

<div align="center">

Built as a portfolio project — feedback and stars ⭐ welcome.

</div>
