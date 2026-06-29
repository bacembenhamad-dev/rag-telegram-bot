"""
RAG chain: semantic retrieval from Qdrant + Groq LLM with per-user memory.
"""

import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ml_book")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "10"))
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
SCORE_THRESHOLD = 0.35

SYSTEM_PROMPT = """You are a friendly, knowledgeable tutor for the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron (2019 edition).

Answer the user's question using ONLY the context excerpts from the book provided below.
- If the context doesn't contain enough information, say: "I couldn't find a clear answer to that in the book." and suggest a related topic the book does cover.
- Be clear and concise. Use short paragraphs, and bullet or numbered lists when they help.
- When relevant, mention the page number from the context (e.g., "As explained on page 142...").
- Do not invent information that is not present in the context.

Formatting (Telegram):
- Use *single asterisks* for bold and _single underscores_ for italics. Never use **double** asterisks.
- Use `backticks` for inline code and triple backticks for code blocks.

Book context:
{context}
"""


class RAGChain:
    def __init__(self):
        print("Loading embedding model...")
        self._embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        print("Connecting to Qdrant...")
        self._qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

        print("Initializing Groq LLM...")
        self._llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1024,
        )

        # {chat_id: [{"role": "user"|"assistant", "content": str}]}
        self._history: dict[int, list[dict]] = defaultdict(list)

    def clear_history(self, chat_id: int) -> None:
        self._history[chat_id] = []

    def _retrieve(self, query: str) -> list[dict]:
        query_vector = self._embedder.embed_query(query)
        results = self._qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=TOP_K,
            score_threshold=SCORE_THRESHOLD,
            with_payload=True,
        )
        return [
            {"text": r.payload["text"], "page": r.payload["page"], "score": r.score}
            for r in results
        ]

    def _build_messages(self, history: list[dict], context: str, question: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    def _format_sources(self, docs: list[dict]) -> str:
        if not docs:
            return ""
        pages = sorted({d["page"] for d in docs})
        page_list = ", ".join(f"p.{p}" for p in pages)
        return f"_Sources: {page_list}_"

    def answer(self, chat_id: int, question: str) -> str:
        docs = self._retrieve(question)

        if not docs:
            no_context = "I couldn't find relevant content in the book for your question. Try rephrasing or ask about a specific ML concept covered in the book."
            self._update_history(chat_id, question, no_context)
            return no_context

        context = "\n\n---\n\n".join(
            f"[Page {d['page']}]\n{d['text']}" for d in docs
        )

        window = self._history[chat_id][-MEMORY_WINDOW:]
        messages = self._build_messages(window, context, question)

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            else:
                lc_messages.append(AIMessage(content=m["content"]))

        response = self._llm.invoke(lc_messages)
        answer_text = response.content.strip()

        sources = self._format_sources(docs)
        full_response = f"{answer_text}\n\n{sources}" if sources else answer_text

        self._update_history(chat_id, question, answer_text)
        return full_response

    def _update_history(self, chat_id: int, question: str, answer_text: str) -> None:
        self._history[chat_id].append({"role": "user", "content": question})
        self._history[chat_id].append({"role": "assistant", "content": answer_text})
        # Keep only the last MEMORY_WINDOW * 2 messages (pairs)
        max_msgs = MEMORY_WINDOW * 2
        if len(self._history[chat_id]) > max_msgs:
            self._history[chat_id] = self._history[chat_id][-max_msgs:]
