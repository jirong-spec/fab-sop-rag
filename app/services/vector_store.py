import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


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


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-init the Chroma embedding model (used for entity extraction)."""
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model, model_kwargs=_cuda_kwargs())


@lru_cache(maxsize=1)
def _get_reranker_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-init the reranker embedding model (used for graph triple scoring).

    Falls back to the Chroma embedding model when RERANKER_MODEL is not set.
    """
    model = settings.reranker_model or settings.embedding_model
    logger.info("Loading reranker model: %s", model)
    return HuggingFaceEmbeddings(model_name=model, model_kwargs=_cuda_kwargs())


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
