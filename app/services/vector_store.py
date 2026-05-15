import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-init the embedding model; shared by vector store and answer reranker."""
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def _get_vector_store() -> Chroma:
    """Lazy-init the vector store; cached after first call."""
    logger.info("Opening Chroma store at: %s", settings.chroma_dir)
    return Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=_get_embeddings(),
        collection_name="sop_docs",
    )


def similarity_search(question: str, k: int = 4) -> list[str]:
    """Return top-k document texts similar to the question."""
    db = _get_vector_store()
    docs = db.similarity_search(question, k=k)
    return [doc.page_content for doc in docs]
