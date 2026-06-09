"""
Ingest SOP markdown documents into the Qdrant vector store.

Usage (inside Docker):
    docker compose run --rm api python scripts/ingest_vector.py

Usage (local, from fab-sop-rag/):
    QDRANT_URL=http://localhost:6333 python scripts/ingest_vector.py

Each .md file in data/sop_docs/ is split into chunks of ~200 characters
with 40-character overlap, then embedded and stored in Qdrant.
The collection name comes from settings.qdrant_collection ("sop_docs") —
the same collection the API's retrieval service reads, so queries
automatically hit this data.

The collection is recreated on every run (force_recreate=True) so that
deleting a source .md file removes its chunks: a plain upsert would leave
orphaned vectors behind for documents that no longer exist.
"""

import logging
import sys
import uuid
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
# Chunk size guided by scripts/eval_chunk_ablation.py under a dev/test split: the prior
# 400/80 baseline was consistently the WORST on both the dev and held-out test questions,
# while ~200-char windows ranked among the best on both. The dev set is tiny (8 questions),
# so this is NOT a statistically significant unique optimum — but 200/40 is a sound default
# that isolates each SOP step into its own chunk (better recall within a context budget).
_CHUNK_SIZE = 200
_CHUNK_OVERLAP = 40
# Stable namespace so re-runs produce identical point IDs (idempotent upserts).
_ID_NAMESPACE = uuid.UUID("a1b2c3d4-0000-4000-8000-000000000001")


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
    from langchain_core.documents import Document
    from langchain_qdrant import QdrantVectorStore

    from app.services.vector_store import _get_embeddings

    md_files = sorted(_DOCS_DIR.glob("*.md"))
    if not md_files:
        logger.error("No .md files found in %s", _DOCS_DIR)
        sys.exit(1)

    logger.info("Found %d SOP documents: %s", len(md_files), [f.name for f in md_files])

    # Reuse the app's embedder so the e5 query/passage prefixes are applied consistently
    # (docs embedded with "passage: "; queries with "query: " at serve time).
    logger.info("Loading embedding model: %s", settings.embedding_model)
    embeddings = _get_embeddings()

    documents: list[Document] = []
    ids: list[str] = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        logger.info("  %s → %d chunks", md_file.name, len(chunks))
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": md_file.name, "chunk_index": i},
                )
            )
            ids.append(str(uuid.uuid5(_ID_NAMESPACE, f"{md_file.stem}__chunk{i:04d}")))

    logger.info("Connecting to Qdrant at %s (collection=%s)", settings.qdrant_url, settings.qdrant_collection)
    # force_recreate=True drops the collection first so removed source files
    # leave no orphaned vectors; vector size + Cosine distance are inferred
    # from the embedding model.
    QdrantVectorStore.from_documents(
        documents,
        embedding=embeddings,
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        ids=ids,
        force_recreate=True,
    )

    logger.info(
        "Vector ingest complete: %d files, %d chunks → collection=%r at %s",
        len(md_files),
        len(documents),
        settings.qdrant_collection,
        settings.qdrant_url,
    )


if __name__ == "__main__":
    main()
