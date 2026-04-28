"""
Ingest SOP markdown documents into the Chroma vector store.

Usage (inside Docker):
    docker compose run --rm api python scripts/ingest_vector.py

Usage (local, from fab-sop-rag/):
    CHROMA_DIR=./chroma_store python scripts/ingest_vector.py

Each .md file in data/sop_docs/ is split into chunks of ~400 characters
with 80-character overlap, then embedded and stored in Chroma.
The collection name is "sop_docs" — same as the one used by the API's
retrieval service so queries automatically hit this data.
"""

import logging
import sys
from pathlib import Path

# Allow `from app.config import settings` when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "sop_docs"
_COLLECTION_NAME = "sop_docs"
_CHUNK_SIZE = 400
_CHUNK_OVERLAP = 80


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def main() -> None:
    from langchain_huggingface import HuggingFaceEmbeddings
    import chromadb

    md_files = sorted(_DOCS_DIR.glob("*.md"))
    if not md_files:
        logger.error("No .md files found in %s", _DOCS_DIR)
        sys.exit(1)

    logger.info("Found %d SOP documents: %s", len(md_files), [f.name for f in md_files])

    logger.info("Loading embedding model: %s", settings.embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    logger.info("Opening Chroma store at: %s", settings.chroma_dir)
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        logger.info("  %s → %d chunks", md_file.name, len(chunks))

        ids = [f"{md_file.stem}__chunk{i:04d}" for i in range(len(chunks))]
        metadatas = [{"source": md_file.name, "chunk_index": i} for i in range(len(chunks))]

        # Embed in one batch per file
        vectors = embeddings.embed_documents(chunks)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    logger.info(
        "Vector ingest complete: %d files, %d chunks → collection=%r at %s",
        len(md_files),
        total_chunks,
        _COLLECTION_NAME,
        settings.chroma_dir,
    )


if __name__ == "__main__":
    main()
