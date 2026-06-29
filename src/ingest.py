"""
One-time ingestion script: PDF → chunks → embeddings → Qdrant Cloud.
Run with: python -m src.ingest
"""

import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ml_book")
PDF_PATH = os.getenv(
    "PDF_PATH",
    "PDFs/2-Aurélien-Géron-Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow_-Concepts-Tools-and-Techniques-to-Build-Intelligent-Systems-O'Reilly-Media-2019.pdf",
)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from each page, returning list of {text, page} dicts."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({"text": text, "page": page_num + 1})
    doc.close()
    print(f"Extracted text from {len(pages)} pages.")
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split page texts into overlapping chunks, preserving page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for item in pages:
        splits = splitter.split_text(item["text"])
        for split in splits:
            chunks.append({"text": split, "page": item["page"]})
    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    return chunks


def embed_and_upsert(chunks: list[dict], client: QdrantClient, embedder: HuggingFaceEmbeddings):
    """Embed chunks in batches and upsert into Qdrant."""
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embedder.embed_documents(texts)
        points = [
            PointStruct(
                id=start + i,
                vector=vectors[i],
                payload={"text": batch[i]["text"], "page": batch[i]["page"]},
            )
            for i in range(len(batch))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        end = min(start + BATCH_SIZE, total)
        print(f"Upserted {end}/{total} chunks...")


def find_pdf(configured_path: str) -> Path:
    """Resolve PDF path; fall back to any PDF in the PDFs/ folder."""
    p = Path(configured_path)
    if p.exists():
        return p
    # Windows encoding issues with accented filenames — search by glob
    candidates = list(Path("PDFs").glob("*.pdf"))
    if candidates:
        found = candidates[0]
        print(f"Note: configured path not found, using '{found.name}' instead.")
        return found
    print(f"ERROR: No PDF found at '{p.resolve()}' or in PDFs/")
    sys.exit(1)


def main():
    pdf_path = find_pdf(PDF_PATH)

    print("Loading embedding model...")
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_size = len(embedder.embed_query("test"))

    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        count = client.count(COLLECTION_NAME).count
        print(f"Collection '{COLLECTION_NAME}' already exists with {count} vectors. Skipping ingestion.")
        print("Delete the collection manually in Qdrant Cloud UI if you want to re-ingest.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}' (dim={vector_size}).")

    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages)
    embed_and_upsert(chunks, client, embedder)

    final_count = client.count(COLLECTION_NAME).count
    print(f"\nDone! Collection '{COLLECTION_NAME}' now has {final_count} vectors.")


if __name__ == "__main__":
    main()
