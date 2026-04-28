"""
Candidate entity extraction from vector-retrieved SOP documents.

Entities are the seeds for graph expansion: they are used as starting nodes
in the Neo4j Cypher query `MATCH p=(n)-[*1..hop]-(m) WHERE n.name IN $ents`.

Entity naming conventions in semiconductor SOP graphs:
  - TitleCase compound words:       EtchStation, VacuumPump, PressureAnomaly
  - Underscore-delimited codes:     SOP_Etch_001, CheckVacuumPump_Step1
  - ALL-CAPS acronyms:              CVD, CMP, ALD, RIE, ICP, PVD, SOP

The three-pattern heuristic in text_utils covers all of these.

MVP note: upgrade to a domain-specific NER model (or exact-match dictionary
against a known node list from Neo4j) for production deployments.
"""

import logging

from app.services.vector_store import similarity_search
from app.utils.text_utils import extract_candidate_tokens

logger = logging.getLogger(__name__)


def extract_candidate_entities(
    question: str,
    k: int = 4,
    max_entities: int = 15,
) -> list[str]:
    """
    Extract candidate SOP entity names from the question itself and vector-retrieved docs.

    Flow:
      1. Extract tokens directly from the question (highest-confidence seeds)
      2. Retrieve top-k documents from Chroma closest to the question
      3. Apply three-pattern token extraction across all documents
      4. Merge, dedup, return up to max_entities unique tokens

    Extracting from the question first ensures that explicit entity references
    (e.g. "SOP_Etch_001", "TurboVacuumPump") are always included even when
    vector retrieval returns documents from a different SOP context.
    """
    seen: set[str] = set()
    result: list[str] = []

    # Step 1: tokens from the question itself (seeds the graph traversal reliably)
    for token in extract_candidate_tokens(question):
        if token not in seen:
            seen.add(token)
            result.append(token)

    # Step 2: augment with tokens from retrieved documents
    docs = similarity_search(question, k=k)
    for doc in docs:
        for token in extract_candidate_tokens(doc):
            if token not in seen:
                seen.add(token)
                result.append(token)

    entities = result[:max_entities]
    logger.debug("Candidate SOP entities: %s", entities)
    return entities
