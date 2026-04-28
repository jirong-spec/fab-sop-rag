"""
Entity token extraction heuristics for semiconductor SOP knowledge queries.

Three complementary patterns handle the naming conventions common in fab documents:

  1. Entity codes   — underscore-separated identifiers: EtchStation_SOP_001, SOP_Etch_001
  2. TitleCase      — compound proper nouns:            EtchStation, VacuumPump, PressureAnomaly
  3. Acronyms       — ALL-CAPS process/equipment terms: CVD, CMP, RIE, ALD, PVD, ICP, RF

Patterns are applied in order from most-specific to least-specific.
Deduplication is done globally so a token found by multiple patterns appears only once.

MVP note: this heuristic works well for node names following common fab naming conventions.
For free-text SOP documents with informal terminology, a domain-specific NER model
(e.g. fine-tuned on semiconductor process corpora) would improve recall significantly.
"""

import re

# Underscore-delimited identifiers starting with a capital letter.
# Captures: EtchStation_SOP_001, SOP_Etch_001, CheckVacuumPump_Step1
_ENTITY_CODE = re.compile(r"[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+")

# CamelCase / TitleCase compound words (≥ 3 chars to skip short abbreviations).
# Captures: EtchStation, VacuumPump, PressureAnomaly, RecipeParameter
_TITLE_CASE = re.compile(r"[A-Z][A-Za-z]{2,}")

# ALL-CAPS acronyms of 2-6 letters common in semiconductor manufacturing.
# Captures: CVD, CMP, SOP, ALD, PVD, ICP, RIE, LPCVD, PECVD
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")


def extract_candidate_tokens(text: str) -> list[str]:
    """
    Extract candidate entity tokens from text using three semiconductor-aware patterns.

    Returns a deduplicated list in insertion order (entity codes first,
    then TitleCase words, then acronyms).
    """
    raw: list[str] = (
        _ENTITY_CODE.findall(text)   # most specific: SOP_001, EtchStation_SOP_001
        + _TITLE_CASE.findall(text)  # compound proper nouns
        + _ACRONYM.findall(text)     # ALL-CAPS acronyms
    )
    seen: set[str] = set()
    result: list[str] = []
    for tok in raw:
        if tok not in seen:
            seen.add(tok)
            result.append(tok)
    return result


# Backward-compatible alias (used by candidate_entities.py)
def extract_uppercase_tokens(text: str) -> list[str]:
    return extract_candidate_tokens(text)
