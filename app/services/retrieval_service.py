import logging

from app.services.vector_store import similarity_search
from app.services.graph_store import graph_expand
from app.utils.text_utils import extract_candidate_tokens

logger = logging.getLogger(__name__)


def _extract_entities(question: str, k: int = 4, max_entities: int = 15) -> list[str]:
    """Extract candidate entity names from question + vector-retrieved docs."""
    seen: set[str] = set()
    result: list[str] = []

    for token in extract_candidate_tokens(question):
        if token not in seen:
            seen.add(token)
            result.append(token)

    for doc in similarity_search(question, k=k):
        for token in extract_candidate_tokens(doc):
            if token not in seen:
                seen.add(token)
                result.append(token)

    return result[:max_entities]


def retrieve(
    question: str,
    top_k: int = 4,
    max_hop: int = 2,
) -> tuple[list[str], list[str]]:
    """
    Full retrieval: question → entity extraction → graph expansion.

    Returns (entities, evidence_triples).
    """
    entities = _extract_entities(question, k=top_k)
    logger.info("Candidate entities: %s", entities)

    triples = graph_expand(entities, hop=max_hop)
    logger.info("Triples retrieved: %d", len(triples))

    return entities, triples
