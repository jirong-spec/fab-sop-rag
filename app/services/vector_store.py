import logging
import threading

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.config import settings

logger = logging.getLogger(__name__)


class _PrefixedEmbeddings(Embeddings):
    """Wrap an embeddings backend, prepending instruction prefixes at EMBED time only.

    e5-family models expect "query: " / "passage: " prefixes; applying them here keeps the
    stored chunk text (page_content) clean while still embedding the prefixed form.
    """

    def __init__(self, base: Embeddings, query_prefix: str, passage_prefix: str):
        self._base = base
        self._q = query_prefix
        self._p = passage_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_documents([self._p + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(self._q + text)


_embeddings: Embeddings | None = None
_reranker_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: QdrantVectorStore | None = None
# Three separate locks to prevent deadlock: _get_vector_store calls _get_embeddings
# while holding its own lock, so a single shared Lock would deadlock.
_embeddings_lock = threading.Lock()
_reranker_lock = threading.Lock()
_vector_store_lock = threading.Lock()


def _cuda_kwargs() -> dict:
    """Return model_kwargs with CUDA device if available, else CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Embedding device: cuda")
            return {"device": "cuda"}
    except ImportError:
        pass
    logger.info("Embedding device: cpu")
    return {"device": "cpu"}


def _get_embeddings() -> Embeddings:
    """Lazy-init the document embedding model; singleton with double-checked locking.

    Wraps the model in _PrefixedEmbeddings when query/passage prefixes are configured
    (e5-family); the doc retrieval embedder and the reranker are separate singletons.
    """
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                logger.info("Loading embedding model: %s", settings.embedding_model)
                base = HuggingFaceEmbeddings(model_name=settings.embedding_model, model_kwargs=_cuda_kwargs())
                if settings.embedding_query_prefix or settings.embedding_passage_prefix:
                    logger.info(
                        "Embedding prefixes: query=%r passage=%r",
                        settings.embedding_query_prefix,
                        settings.embedding_passage_prefix,
                    )
                    _embeddings = _PrefixedEmbeddings(
                        base, settings.embedding_query_prefix, settings.embedding_passage_prefix
                    )
                else:
                    _embeddings = base
    return _embeddings


def _get_reranker_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-init the reranker embedding model; singleton with double-checked locking.

    Falls back to the document embedding model when RERANKER_MODEL is not set.
    """
    global _reranker_embeddings
    if _reranker_embeddings is None:
        with _reranker_lock:
            if _reranker_embeddings is None:
                model = settings.reranker_model or settings.embedding_model
                logger.info("Loading reranker model: %s", model)
                _reranker_embeddings = HuggingFaceEmbeddings(model_name=model, model_kwargs=_cuda_kwargs())
    return _reranker_embeddings


def _get_vector_store() -> QdrantVectorStore:
    """Lazy-init the Qdrant vector store; singleton with double-checked locking.

    Connects to an existing collection (created by scripts/ingest_vector.py).
    Raises if the collection does not exist yet — callers (warmup, health probe)
    wrap this in try/except so a not-yet-ingested store degrades gracefully.
    """
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                logger.info(
                    "Connecting to Qdrant at %s (collection=%s)",
                    settings.qdrant_url,
                    settings.qdrant_collection,
                )
                _vector_store = QdrantVectorStore.from_existing_collection(
                    embedding=_get_embeddings(),
                    collection_name=settings.qdrant_collection,
                    url=settings.qdrant_url,
                )
    return _vector_store


def similarity_search(question: str, k: int = 4) -> list[str]:
    """Return top-k document texts similar to the question."""
    db = _get_vector_store()
    docs = db.similarity_search(question, k=k)
    return [doc.page_content for doc in docs]
