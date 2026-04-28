import logging

from app.services.candidate_entities import extract_candidate_entities
from app.services.graph_store import graph_expand

logger = logging.getLogger(__name__)


def retrieve(
    question: str,
    top_k: int = 4,
    max_hop: int = 2,
) -> tuple[list[str], list[str]]:
    """
    Full retrieval pipeline: vector → candidate entities → graph expansion.

    Returns:
        candidate_entities: entity names extracted from vector documents
        evidence_triples:   graph triples expanded from those entities
    """
    entities = extract_candidate_entities(question, k=top_k)
    logger.info("Candidate entities: %s", entities)

    triples = graph_expand(entities, hop=max_hop)
    logger.info("Triples retrieved: %d", len(triples))

    return entities, triples
